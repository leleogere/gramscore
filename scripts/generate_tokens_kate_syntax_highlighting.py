"""
Generate a Kate syntex-highlighting file for tokens using a list of colors.
Each color corresponds to a voice index.

<?xml version="1.0" encoding="UTF-8"?>
<language version="1.0" kateversion="5.1" name="Tokens" extensions="*.tokens">
    <highlighting>
        <contexts>
            <context lineEndContext="#pop">
                <RegExpr attribute="VoiceIdx0" context="#pop" String=".*voice_idx=i.*" />
            </context>
        </contexts>
        <itemDatas>
            <itemData name="Normal text" bold="1" />
            <itemData name="VoiceIdxi" color="colors[i]" />
        </itemDatas>
    </highlighting>
</language>
"""

from lxml import etree
from pathlib import Path


def generate_xml(colors: list[str]) -> etree.Element:
    # Validate colors list
    for i, color in enumerate(colors):
        if not color.startswith("#"):
            colors[i] = "#" + color
        if not len(color) == 7:
            raise ValueError(f"Invalid color: {color}. Must be in the format #RRGGBB")

    # Create root element
    root = etree.Element("language")
    root.set("name", "Automata Tokens")
    root.set("version", "1.0")
    root.set("kateversion", "5.1")
    root.set("extensions", "*.tokens")

    # Add <highlighting> element
    highlighting = etree.SubElement(root, "highlighting")

    # Add <contexts> element
    contexts = etree.SubElement(highlighting, "contexts")
    context = etree.SubElement(contexts, "context")

    # Add <itemDatas> element and default <itemData> element
    item_datas = etree.SubElement(highlighting, "itemDatas")
    etree.SubElement(item_datas, "itemData")  # Needed for some reason?

    # Create <RegExpr> and <itemData> elements
    for i, color in enumerate(colors):
        reg_expr = etree.SubElement(context, "RegExpr")
        reg_expr.set("attribute", f"VoiceIdx{i}")
        reg_expr.set("String", f".*voice_idx={i}.*")

        item_data = etree.SubElement(item_datas, "itemData")
        item_data.set("name", f"VoiceIdx{i}")
        item_data.set("color", color)
        item_data.set("selColor", color)

    # Add Bar token
    reg_expr = etree.SubElement(context, "RegExpr")
    reg_expr.set("attribute", "Bar")
    reg_expr.set("String", r"Bar\(.*")
    reg_expr.set("beginRegion", "Bar")
    reg_expr.set("endRegion", "Bar")
    item_data = etree.SubElement(item_datas, "itemData")
    item_data.set("name", "Bar")
    item_data.set("bold", "true")

    return root


if __name__ == '__main__':
    output_path = Path("/home/gerel/.local/share/org.kde.syntax-highlighting/syntax/tokens.xml")
    colors = "#000FB3,#BD1C00,#00C750,#8800D1,#DBC200,#00C7E6,#F00094,#60FA00,#0526FF,#FF270F,#1AFF6A,#A724FF,#FFDC2E,#38EEFF,#FF42C0,#9AFF4D"
    colors = colors.split(",")
    root = generate_xml(colors)
    with open(output_path, "wb") as f:
        f.write(etree.tostring(root, pretty_print=True))
