Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 6 items.

=== ITEM id=1aa02c6f43a358a2 ===
Gold sentences and facts:
Sentence 0: Sanitary napkins ( also called sanitary towels or pads ) -- Rectangular pieces of material worn attached to the underwear to absorb menstrual flow , often with an adhesive backing to hold the pad in place . Disposable pads may contain wood pulp or gel products , usually with a plastic lining and bleached . Tampons -- Disposable cylinders of treated rayon / cotton blends or all - cotton fleece , usually bleached , that are inserted into the vagina to absorb menstrual flow . Disposable menstrual cups made of soft plastic -- A firm , flexible cup - shaped device worn inside the vagina to collect menstrual flow .
  Fact 0.0: Sanitary napkins are also called sanitary towels or pads.
  Fact 0.1: Sanitary napkins are rectangular pieces of material worn attached to the underwear to absorb menstrual flow.
  Fact 0.2: Sanitary napkins are often given an adhesive backing to hold the pad in place.
  Fact 0.3: Disposable sanitary pads may contain wood pulp or gel products.
  Fact 0.4: Disposable sanitary pads usually have a plastic lining and are bleached.
  Fact 0.5: Tampons are disposable cylinders of treated rayon / cotton blends or all-cotton fleece.
  Fact 0.6: Tampons are usually bleached.
  Fact 0.7: Tampons are inserted into the vagina to absorb menstrual flow.
  Fact 0.8: Disposable menstrual cups are made of soft plastic.
  Fact 0.9: Disposable menstrual cups are a firm, flexible, cup-shaped device.
  Fact 0.10: Disposable menstrual cups are worn inside the vagina.
  Fact 0.11: Disposable menstrual cups collect menstrual flow.
Sentence 1: Disposable menstrual cups made of soft plastic -- A firm , flexible cup - shaped device worn inside the vagina to collect menstrual flow .
  Fact 1.0: Disposable menstrual cups are made of soft plastic.
  Fact 1.1: Disposable menstrual cups are a firm, flexible, cup-shaped device.
  Fact 1.2: Disposable menstrual cups are worn inside the vagina.
  Fact 1.3: Disposable menstrual cups collect menstrual flow.
Sentence 2: Menstrual cups -- A firm , flexible bell - shaped device worn inside the vagina to collect menstrual flow . Menstrual cups are usually made of silicone and can last 5 years or longer .
  Fact 2.0: Menstrual cups are a firm, flexible, bell-shaped device.
  Fact 2.1: Menstrual cups are worn inside the vagina.
  Fact 2.2: Menstrual cups collect menstrual flow.
  Fact 2.3: Menstrual cups are usually made of silicone.
  Fact 2.4: Menstrual cups can last 5 years or longer.

Retrieved items:
<item 1>
Graph query row:
entity: Menstrual cup
relationship: PART_OF
related: Reusable products
</item 1>

<item 2>
Menstruation
Instead , they use materials found in the environment or other improvised materials . Absorption materials that may be used by women who can not afford anything
else include : sand , ash , small hole in earth , cloth , whole leaf , leaf fibre ( such as water hyacinth , banana , papyrus , cotton fibre ) , paper ( toilet
paper , re-used newspaper , pulped and dried paper ) , animal pelt e.g. goat skin , double layer of underwear , skirt or sari . Society and culture Amra Padatik
India , celebration of Menstrual Hygiene Day in India Traditions and taboos Main article : Culture and menstruation Further information : Menstrual taboo Many
religions have menstruation - related traditions . Islam Islam prohibits sexual contact with women during menstruation in the 2nd chapter of the Quran . Women
are exempted from Salah ( prayer ) . They are also prohibited from fasting and obligatory fasts of Ramadan are to be performed during other days . During menses
pilgrimages are allowed ; however , circumambulation of the Kaaba is prohibited and is to be performed during other times .
</item 2>

<item 3>
Menstruation
Some anthropologists have questioned the energy cost of rebuilding the endometrium every fertility cycle . However , anthropologist Beverly Strassmann has
proposed that the energy savings of not having to continuously maintain the uterine lining more than offsets energy cost of having to rebuild the lining in the
next fertility cycle , even in species such as humans where much of the lining is lost through bleeding ( overt menstruation ) rather than reabsorbed ( covert
menstruation ) . Many have questioned the evolution of overt menstruation in humans and related species , speculating on what advantage there could be to losing
blood associated with dismantling the endometrium , rather than absorbing it , as most mammals do . Humans do , in fact , reabsorb about two - thirds of the
endometrium each cycle . Strassmann asserts that overt menstruation occurs not because it is beneficial in itself . Rather , the fetal development of these
species requires a more developed endometrium , one which is too thick to reabsorb completely . Strassman correlates species that have overt menstruation to
those that have a large uterus relative to the adult female body size .
</item 3>

<item 4>
Menstruation
The intention behind this decision was the hope of the inventor , John Rock , to win approval for his invention from the Roman Catholic Church . That attempt
failed , but the 28 - day cycle remained the standard when the pill became available to the public . There is debate among medical researchers about the
potential long - term impacts of these practices upon female health . Some researchers point to the fact that historically , females have had far fewer
menstrual periods throughout their lifetimes , a result of shorter life expectancies , as well as a greater length of time spent pregnant or breast - feeding ,
which reduced the number of periods experienced by females . These researchers believe that the higher number of menstrual periods by females in modern
societies may have a negative impact upon their health . On the other hand , some researchers believe there is a greater potential for negative impacts from
exposing females perhaps unnecessarily to regular low doses of synthetic hormones over their reproductive years . Breastfeeding Main article : Lactational
amenorrhea method Breastfeeding causes negative feedback to occur on pulse secretion of gonadotropin - releasing hormone ( GnRH ) and luteinizing hormone ( LH )
.
</item 4>

<item 5>
Menstruation
The egg is released from an ovary around day fourteen in the cycle ; the thickened lining of the uterus provides nutrients to an embryo after implantation . If
pregnancy does not occur , the lining is released in what is known as menstruation . Contents ( hide ) 1 Onset and frequency 2 Health effects 2.1 Cramps 2.2
Mood and behavior 2.3 Bleeding 2.4 Menstrual disorders 3 Ovulation suppression 3.1 Birth control 3.2 Breastfeeding 4 Menstrual management 4.1 Disposable
products 4.2 Reusable products 4.3 Non-commercial items 5 Society and culture 5.1 Traditions and taboos 5.1. 1 Islam 5.1. 2 Judaism 5.1. 3 Hinduism 5.1. 4
Christianity 5.1. 5 Secular culture 5.1.
</item 5>

<item 6>
Menstruation
Many now argue to describe menstruation in simple terminology , including : Cycle regularity ( irregular , regular , or absent ) Frequency of menstruation (
frequent , normal , or infrequent ) Duration of menstrual flow ( prolonged , normal , or shortened ) Volume of menstrual flow ( heavy , normal , or light )
Dysfunctional uterine bleeding is a hormonally caused bleeding abnormality . Dysfunctional uterine bleeding typically occurs in premenopausal women who do not
ovulate normally ( i.e. are anovulatory ) . All these bleeding abnormalities need medical attention ; they may indicate hormone imbalances , uterine fibroids ,
or other problems . As pregnant women may bleed , a pregnancy test forms part of the evaluation of abnormal bleeding . Women who had undergone female genital
mutilation ( particularly type III - infibulation ) a practice common in parts of Africa , may experience menstrual problems , such as slow and painful
menstruation , that is caused by the near - complete sealing off of the vagina .
</item 6>

<item 7>
Menstruation
Menstrual cups are usually made of silicone and can last 5 years or longer . Reusable cloth pads -- Pads that are made of cotton ( often organic ) , terrycloth
, or flannel , and may be handsewn ( from material or reused old clothes and towels ) or storebought . Padded panties or period - proof underwear -- Reusable
cloth ( usually cotton ) underwear with extra absorbent layers sewn in to absorb flow . Some also use patented technology to be leak resistant , such as the
brand THINX . Sea sponges -- Natural sponges , worn internally like a tampon to absorb menstrual flow . Blanket , towel -- ( also known as a draw sheet ) --
large reusable piece of cloth , most often used at night , placed between legs to absorb menstrual flow . Non-commercial items In developing countries , many
poor women can not afford commercial feminine hygiene products . Instead , they use materials found in the environment or other improvised materials .
</item 7>

<item 8>
Menstruation
Scotland is the first country to ban period poverty . In 2015 a petition was put forward in England seeking the elimination of the 5 % `` tampon tax '' . The
government voted in favour but it has yet to be implemented due to Brexit complications . In England , one in ten women between 14 and 21 can not afford
menstrual management products . In the United States , nine states have eliminated the tampon tax , and seven states have introduced legislation . In January
2018 , California rejected a proposal to eliminate tampon tax . Evolution All female placental mammals have a uterine lining that builds up when the animal is
fertile , but it is dismantled when the animal is infertile . Most female mammals have an estrous cycle , yet only primates ( including humans ) , several
species of bats , and elephant shrews have a menstrual cycle . ( Killer whales , Orcinus orca , are one of the few species to experience menopause . ) Some
anthropologists have questioned the energy cost of rebuilding the endometrium every fertility cycle .
</item 8>

<item 9>
Menstruation
Menstruation : A Cultural History . Palgrave Macmillan . ISBN 1 - 4039 - 3935 - 7 . Retrieved 9 November 2013 . Knight , Chris ( 1995 ) . Blood Relations :
Menstruation and the Origins of Culture . New Haven and London : Yale University . ISBN 0 - 300 - 04911 - 0 . Retrieved 9 November 2013 . External links
Wikiquote has quotations related to : Menstruation Wikimedia Commons has media related to Menstruation . Museum of Menstruation Human physiology of sexual
reproduction Menstrual cycle Menarche Menstruation Follicular phase Ovulation Luteal phase Gametogenesis Spermatogenesis ( spermatogonium spermatocyte spermatid
sperm ) Oogenesis ( oogonium oocyte ootid ovum ) Germ cell ( gonocyte gamete ) Human sexual behavior Sexual arousal Sexual intercourse Masturbation Erection
Orgasm Ejaculation Insemination Fertilisation / Fertility Implantation Pregnancy Postpartum period Mechanics of sex Life span Prenatal development / Sexual
dimorphism / Sexual differentiation ( Feminization Virilization
</item 9>

<item 10>
Menstruation
Menstrual products ( also called `` feminine hygiene '' products ) are made to absorb or catch menstrual blood . A number of different products are available -
some are disposable , some are reusable . Where women can afford it , items used to absorb or catch menses are usually commercially manufactured products .
Disposable products Disposable sanitary napkin Tampon in plastic applicator Sanitary napkins ( also called sanitary towels or pads ) -- Rectangular pieces of
material worn attached to the underwear to absorb menstrual flow , often with an adhesive backing to hold the pad in place . Disposable pads may contain wood
pulp or gel products , usually with a plastic lining and bleached . Tampons -- Disposable cylinders of treated rayon / cotton blends or all - cotton fleece ,
usually bleached , that are inserted into the vagina to absorb menstrual flow . Disposable menstrual cups made of soft plastic -- A firm , flexible cup - shaped
device worn inside the vagina to collect menstrual flow . Reusable products Menstrual cup Cloth menstrual pad Menstrual cups -- A firm , flexible bell - shaped
device worn inside the vagina to collect menstrual flow . Menstrual cups are usually made of silicone and can last 5 years or longer .
</item 10>
=== END ITEM id=1aa02c6f43a358a2 ===

=== ITEM id=e49f21e4703ed1f1 ===
Gold sentences and facts:
Sentence 0: Cuban War of Independence An estimated 200,000 Cubans died in Spanish concentration camps . Date February 24 , 1895 -- February 15 , 1898 ( 2 years , 11 months , 3 weeks and 1 day ) Location Cuba ( Caribbean ) Result Cuban victory American Intervention ; Expulsion of the Spanish colonial government during Spanish -- American War ( 1898 ) . Treaty of Paris United States Military Government in Cuba Cuban independence Belligerents Spain Spanish Cuba Spanish Philippines Cuban nationalists United States Commanders and leaders Arsenio Linares Manuel Macías Ramón Blanco Valeriano Weyler Patricio Montojo Pascual Cervera Máximo Gómez Calixto García José Martí † Antonio Maceo † Nelson A. Miles William Shafter George Dewey William T. Sampson Strength 196,000 53,774 rebels 28,000 -- 60,000 US regulars and 200,000 + volunteers Casualties and losses 786 killed in action 8,627 DOW 53,313 dead from disease 900 killed in action ( 1898 ) 5,180 Cuban insurgents died in combat 3,437 Cuban insurgents dead from disease 385 US soldiers died in combat 2,061 US soldiers dead from disease 300,000 total Cubans dead ( most of them civilian fatalities in the concentration camps )
  Fact 0.0: An estimated 200,000 Cubans died in Spanish concentration camps.
  Fact 0.1: The Cuban War of Independence started on February 24, 1895.
  Fact 0.2: The Cuban War of Independence ended on February 15, 1898.
  Fact 0.3: The Cuban War of Independence lasted 2 years, 11 months, 3 weeks and 1 day.
  Fact 0.4: The Cuban War of Independence took place in Cuba in the Caribbean.
  Fact 0.5: The result of the Cuban War of Independence was a Cuban victory.
  Fact 0.6: The result of the Cuban War of Independence included American intervention.
  Fact 0.7: The result of the Cuban War of Independence included the expulsion of the Spanish colonial government.
  Fact 0.8: The expulsion of the Spanish colonial government happened during the Spanish-American War (1898).
  Fact 0.9: The Treaty of Paris is listed as a result.
  Fact 0.10: The United States Military Government in Cuba is listed as a result.
  Fact 0.11: Cuban independence is listed as a result.
  Fact 0.12: The belligerents were Spain, Spanish Cuba, Spanish Philippines, Cuban nationalists and the United States.
  Fact 0.13: The commanders and leaders listed include Arsenio Linares, Manuel Macias, Ramon Blanco, Valeriano Weyler, Patricio Montojo and Pascual Cervera.
  Fact 0.14: The commanders and leaders listed include Maximo Gomez, Calixto Garcia, Jose Marti (died) and Antonio Maceo (died).
  Fact 0.15: The commanders and leaders listed include Nelson A. Miles, William Shafter, George Dewey and William T. Sampson.
  Fact 0.16: The strength figures include 196,000 and 53,774 rebels.
  Fact 0.17: The strength figures include 28,000 to 60,000 US regulars.
  Fact 0.18: The strength figures include 200,000 or more volunteers.
  Fact 0.19: 786 were killed in action.
  Fact 0.20: 8,627 died of wounds.
  Fact 0.21: 53,313 died from disease.
  Fact 0.22: 900 were killed in action in 1898.
  Fact 0.23: 5,180 Cuban insurgents died in combat.
  Fact 0.24: 3,437 Cuban insurgents died from disease.
  Fact 0.25: 385 US soldiers died in combat.
  Fact 0.26: 2,061 US soldiers died from disease.
  Fact 0.27: The total of Cubans dead was 300,000.
  Fact 0.28: Most of the 300,000 Cubans dead were civilian fatalities in the concentration camps.
Sentence 1: Martí was killed shortly after landing on May 19 , 1895 , at Dos Rios , but Máximo Gomez and Antonio Maceo fought on , taking the war to all parts of Oriente . By the end of June , all of Camagüey was at war . Based on new research in Cuban sources , historian John Lawrence Tone demonstrated that Gomez and Maceo were the first to force the civilian forces to choose sides . `` Either they relocated to the east side of the islands , where the Cubans controlled the mountainous terrain , or they would be accused of supporting the Spanish and be subject to immediate trial and execution . '' Continuing west , they were joined by 1868 war veterans , such as Polish internationalist General Carlos Roloff and Serafín Sánchez in Las Villas , who brought weapons , men and experience to the revolutionaries ' arsenal .
  Fact 1.0: Jose Marti was killed shortly after landing on May 19, 1895.
  Fact 1.1: Jose Marti was killed at Dos Rios.
  Fact 1.2: Maximo Gomez fought on after Marti was killed.
  Fact 1.3: Antonio Maceo fought on after Marti was killed.
  Fact 1.4: Gomez and Maceo took the war to all parts of Oriente.
  Fact 1.5: By the end of June, all of Camaguey was at war.
  Fact 1.6: Historian John Lawrence Tone demonstrated, based on new research in Cuban sources, that Gomez and Maceo were the first to force the civilian forces to choose sides.
  Fact 1.7: Civilians had to relocate to the east side of the islands, where the Cubans controlled the mountainous terrain, or be accused of supporting the Spanish.
  Fact 1.8: Civilians who were accused of supporting the Spanish would be subject to immediate trial and execution.
  Fact 1.9: Gomez and Maceo continued west.
  Fact 1.10: Gomez and Maceo were joined by 1868 war veterans.
  Fact 1.11: Polish internationalist General Carlos Roloff was one of the 1868 war veterans who joined them.
  Fact 1.12: Serafin Sanchez was one of the 1868 war veterans who joined them in Las Villas.
  Fact 1.13: The 1868 war veterans brought weapons, men and experience to the revolutionaries' arsenal.
Sentence 2: During the years 1869 -- 1888 of the so - called `` Rewarding Truce '' , lasting for 17 years from the end of the Ten Years ' War in 1878 , there were fundamental social changes in Cuban society . With the abolition of slavery in October 1886 , freedmen joined the ranks of farmers and the urban working class . The economy could no longer sustain itself with the shift and changes ; therefore , many wealthy Cubans lost their property , and joined the urban middle class . The number of sugar mills dropped and efficiency increased : only companies , and the most powerful plantation owners , remained in business followed by the Central Board of Artisans in 1879 , and many more across the island . After his second deportation to Spain in 1878 , José Martí moved to the United States in 1881 . There he mobilized the support of the Cuban exile community , especially in Ybor City ( Tampa area ) and Key West , Florida . His goal was revolution in order to achieve independence from Spain . Martí lobbied against the U.S. annexation of Cuba , which was desired by some politicians in both the U.S. and Cuba .
  Fact 2.0: During the years 1869 to 1888, the so-called Rewarding Truce took place.
  Fact 2.1: The Rewarding Truce lasted 17 years.
  Fact 2.2: The Rewarding Truce lasted from the end of the Ten Years' War in 1878.
  Fact 2.3: There were fundamental social changes in Cuban society during the Rewarding Truce.
  Fact 2.4: Slavery was abolished in October 1886.
  Fact 2.5: Freedmen joined the ranks of farmers and the urban working class after the abolition of slavery.
  Fact 2.6: The Cuban economy could no longer sustain itself with the shift and changes.
  Fact 2.7: Many wealthy Cubans lost their property.
  Fact 2.8: Many wealthy Cubans joined the urban middle class.
  Fact 2.9: The number of sugar mills dropped and efficiency increased.
  Fact 2.10: Only companies and the most powerful plantation owners remained in business.
  Fact 2.11: The Central Board of Artisans was formed in 1879.
  Fact 2.12: Many more such groups followed across the island.
  Fact 2.13: Jose Marti was deported to Spain a second time in 1878.
  Fact 2.14: Jose Marti moved to the United States in 1881.
  Fact 2.15: In the United States, Marti mobilized support from the Cuban exile community.
  Fact 2.16: Marti mobilized support especially in Ybor City (Tampa area) and Key West, Florida.
  Fact 2.17: Marti's goal was revolution to achieve independence from Spain.
  Fact 2.18: Marti lobbied against the U.S. annexation of Cuba.
  Fact 2.19: Some politicians in both the U.S. and Cuba desired the U.S. annexation of Cuba.

Retrieved items:
<item 1>
Cuban War of Independence
In mid-September , representatives of the five Liberation Army Corps assembled in Jimaguayú , Camagüey to approve the `` Jimaguayú Constitution . '' They
established a central government , which grouped the executive and legislative powers into one entity named `` Government Council '' , headed by Salvador
Cisneros and Bartolomé Masó . After some time of consolidation in the three eastern provinces , the liberation armies headed for Camagüey and then Matanzas ,
outmaneuvering and deceiving the Spanish Army several times . They defeated Spanish Gen. Arsenio Martínez - Campos y Antón , who had gained victory in the Ten -
Year War , and killed his most trusted general at Peralejo . Campos tried the same strategy he had employed in the Ten Years ' War , constructing a broad belt
across the island , called the trocha , about 80 km long and 200 m wide . This defense line was to confine rebel activities to the eastern provinces . The belt
was developed along a railroad from Jucaro in the south to Morón in the north .
</item 1>

<item 2>
Cuban War of Independence
Contents ( hide ) 1 Background 2 War 3 The Maine incident 4 The Spanish -- American War 5 Peace 6 Analysis 7 See also 8 References 9 Further reading Background
During the years 1869 -- 1888 of the so - called `` Rewarding Truce '' , lasting for 17 years from the end of the Ten Years ' War in 1878 , there were
fundamental social changes in Cuban society . With the abolition of slavery in October 1886 , freedmen joined the ranks of farmers and the urban working class .
The economy could no longer sustain itself with the shift and changes ; therefore , many wealthy Cubans lost their property , and joined the urban middle class
. The number of sugar mills dropped and efficiency increased : only companies , and the most powerful plantation owners , remained in business followed by the
Central Board of Artisans in 1879 , and many more across the island . After his second deportation to Spain in 1878 , José Martí moved to the United States in
1881 . There he mobilized the support of the Cuban exile community , especially in Ybor City ( Tampa area ) and Key West , Florida .
</item 2>

<item 3>
Cuban War of Independence
As the war went on , the major obstacle to Cuban success was weapons supply . Although weapons and funding came from within the Cuban exile community and
supporters in the United States , the supply operation violated American laws . These were enforced by the U.S. Coast Guard : of 71 attempted re-supply missions
, only 27 got through ; 5 were stopped by the Spanish , but 33 were stopped by the U.S. Coast Guard . In 1897 , the liberation army maintained a privileged
position in Camagüey and Oriente , where the Spanish controlled only a few cities . Spanish Liberal leader Práxedes Mateo Sagasta admitted in May 1897 : ``
After having sent 200,000 men and shed so much blood , we do n't own more land on the island than what our soldiers are stepping on '' . The rebel force of
3,000 defeated the Spanish in various encounters , such as the Battle of La Reforma , and forcing the surrender of Las Tunas on August 30 . The Spaniards were
kept on the defensive . Las Tunas had been guarded by over 1,000 well - armed and well - supplied men .
</item 3>

<item 4>
Cuban War of Independence
Blaine 's vision did not allow the existence of an independent Cuba . `` Martí noticed with alarm the movement to annex Hawaii , viewing it as establishing a
pattern for Cuba ... '' War This section does not cite any sources . Please help improve this section by adding citations to reliable sources . Unsourced
material may be challenged and removed . ( February 2012 ) ( Learn how and when to remove this template message ) On December 25 , 1894 , three ships ; the
Lagonda , the Almadis and the Baracoa , set sail for Cuba from Fernandina Beach , Florida , loaded with soldiers and weapons . Two of the ships were seized by
US authorities in early January , but the proceedings went ahead . Not to be dissuaded , on March 25 , Martí presented the Manifesto of Montecristi , which
outlined the policy for Cuba 's war of independence : The war was to be waged by blacks and whites alike ; Participation of all blacks was crucial for victory ;
Spaniards who did not object to the war effort should be spared , Private rural properties should not be damaged ; and The revolution should bring new economic
life to Cuba .
</item 4>

<item 5>
Cuban War of Independence
As stipulated at the Jimaguayü Assembly two years earlier , a second Constituent Assembly met in La Yaya , Camagüey , on October 10 , 1897 . The newly adopted
constitution provided that military command was to be subordinated to civilian rule . The government was confirmed , naming Bartolomé Masó President and Domingo
Méndez Capote Vice President . Madrid decided to change its policy towards Cuba , and replaced Weyler . It also drew up a colonial constitution for Cuba and
Puerto Rico , and installed a new government in Havana . But with half the country out of its control and the other half in arms , the colonial government was
powerless and these changes were rejected by the rebels . The Maine incident Main article : USS Maine ( ACR - 1 ) Wreckage of the Maine , 1898 The Cuban
struggle for independence had captured the American imagination for years . Some newspapers had agitated for US intervention , especially because of its large
financial investment , and featured sensational stories of Spanish atrocities against the native Cuban population , which were exaggerated for propaganda . Such
coverage continued after Spain had replaced Weyler and changed its policies .
</item 5>

<item 6>
Cuban War of Independence
When Cuba 's first war of independence broke out in 1868 , some of those same soldiers were assigned to Cuba ; they brought along what by then become a
derogatory Spanish slur for insurgents . The Cubans adopted the name with pride . After the Ten Years ' War , the government had forbidden possession of weapons
by private individuals . From the beginning of the uprising , the rebels were hampered by the lack of suitable weapons . They compensated by using guerrilla -
style fighting , based on quick raids and fades to the environment , the element of surprise , mounting their forces on fast horses , and using machetes against
regular troops on the march . They acquired most of their weapons and ammunition in raids on the Spaniards . Between June 11 , 1895 , and November 30 , 1897 ,
out of sixty attempts to bring weapons and supplies to the rebels from outside the country , only one succeeded through the protection of the British . Twenty -
eight ships were intercepted within U.S. territory ; five were intercepted at sea by the U.S.
</item 6>

<item 7>
Cuban War of Independence
His goal was revolution in order to achieve independence from Spain . Martí lobbied against the U.S. annexation of Cuba , which was desired by some politicians
in both the U.S. and Cuba . After deliberations with patriotic clubs across the United States , the Antilles and Latin America , `` El Partido Revolucionario
Cubano '' ( The Cuban Revolutionary Party ) was in a state of pendency and was affected by a growing fear that the U.S. government would try to annex Cuba
before the revolution could liberate the island from Spain '' . A new trend of aggressive U.S. `` influence '' was expressed by Secretary of State James G.
Blaine 's suggestion that all of Central and South America would some day fall to the U.S. : `` That rich island '' , Blaine wrote on 1 December 1881 , `` the
key to the Gulf of Mexico , is , though in the hands of Spain , a part of the American commercial system ... If ever ceasing to be Spanish , Cuba must
necessarily become American and not fall under any other European domination '' . Blaine 's vision did not allow the existence of an independent Cuba .
</item 7>

<item 8>
Cuban War of Independence
Around that time , Spanish forces in Cuba numbered about 80,000 , of which 20,000 were regular troops and 60,000 were Spanish and Cuban volunteer militia . The
latter were a locally enlisted force that took care of most of the `` guard and police '' duties on the island . Wealthy landowners would `` volunteer '' a
number of their slaves to serve in this force , which was under local control as militia and not under official military command . By December , Spain had sent
98,412 regular troops to the island , and the colonial government increased the Volunteer Corps to 63,000 men . By the end of 1897 , there were 240,000 regulars
and 60,000 irregulars on the island . The revolutionaries were far outnumbered . The Mambises were named after the Negro Spanish officer , Juan Ethninius Mamby
who joined the Dominican fight for independence in 1844 . The Spanish soldiers referred to the insurgents in the Dominican Republic as `` the men of Mamby '' or
`` Mambies '' .
</item 8>

<item 9>
Cuban War of Independence
The insurrection began on February 24 , 1895 , with uprisings all across the island . In Oriente , the most important ones took place in Santiago , Guantánamo ,
Jiguaní , San Luis , El Cobre , El Caney , Alto Songo , Bayate , and Baire . The uprisings in the central part of the island , such as Ibarra , Jagüey Grande ,
and Aguada , suffered from poor coordination and failed ; the leaders were captured , some of them deported and some executed . In the province of Havana , the
insurrection was discovered before it was underway , and authorities detained its leaders . The insurgents further west in Pinar del Río were ordered by rebel
leaders to wait . On April 1 and 11 , 1895 , the main Mambi leaders landed on two expeditions in Oriente : Major General Antonio Maceo along with 22 members
near Baracoa , and José Martí , Máximo Gómez and 4 other members in Playitas .
</item 9>

<item 10>
Cuban War of Independence
On December 10 , 1898 , the United States and Spain signed the Treaty of Paris , which recognized Cuban independence . Although the Cubans had participated in
the liberation efforts , the United States prevented Cuba from participating in the Paris peace talks and signing the treaty . The treaty set no time limit for
U.S. occupation and the Isle of Pines was excluded from Cuba . The treaty officially granted Cuba 's independence , but U.S. General William R. Shafter refused
to allow Cuban General Calixto García and his rebel forces to participate in the surrender ceremonies in Santiago de Cuba . Analysis Most Spanish historians
argue that the insurgency could not have succeeded on its own . Cuban historians argue the Cubans were on the verge of winning in 1898 and needed no outside
help . See also José Semidei Rodríguez Juan Ríus Rivera References Jump up ^ `` Liberty , Equality , Power : A History of the American People , Volume II :
Since 1863 , Concise Edition '' ^ Jump up to : Clodfelter ( 2017 ) .
</item 10>
=== END ITEM id=e49f21e4703ed1f1 ===

=== ITEM id=cd62d14ec8976535 ===
Gold sentences and facts:
Sentence 0: Bruce Willis as John McClane Jai Courtney as John `` Jack '' McClane , Jr . Sebastian Koch as Yuri Komarov Yuliya Snigir as Irina Komarova Sergei Kolesnikov as Viktor Chagarin Radivoje Bukvić as Alik Mary Elizabeth Winstead as Lucy McClane
  Fact 0.0: Bruce Willis plays John McClane.
  Fact 0.1: Jai Courtney plays John 'Jack' McClane, Jr.
  Fact 0.2: Sebastian Koch plays Yuri Komarov.
  Fact 0.3: Yuliya Snigir plays Irina Komarova.
  Fact 0.4: Sergei Kolesnikov plays Viktor Chagarin.
  Fact 0.5: Radivoje Bukvić plays Alik.
  Fact 0.6: Mary Elizabeth Winstead plays Lucy McClane.

Retrieved items:
<item 1>
A Good Day to Die Hard
Roman Luknár makes an apperance as Anton , a business associate of Chagarin . Ganxsta Zolee makes a brief cameo as a MRAP driver . Production ( edit )
Pre-production ( edit ) Production was formally announced in 2010 , with X-Men Origins : Wolverine and The A-Team writer Skip Woods confirmed as the film 's
screenwriter . Noam Murro was originally attached to direct the film but left production to direct the 300 prequel , 300 : Rise of an Empire . John Moore was
subsequently drafted in to replace him . The film was originally titled `` Die Hard 24 / 7 '' . The media speculated that the film would be a crossover between
the Die Hard and 24 series , with Kiefer Sutherland to reprise his role as Jack Bauer alongside John McClane . This was never confirmed by the studio , and the
film 's title was later revealed to be A Good Day to Die Hard -- with no further mention of any involvement from the 24 series -- with a release date of
February 14 , 2013 .
</item 1>

<item 2>
A Good Day to Die Hard
A Good Day to Die Hard was a box office success but received overwhelmingly negative reviews for its implausible action sequences , cinematography , weak plot ,
cliched script , and lack of characterization , although the special effects were praised . It is the first film produced by TSG Entertainment , since
distributor 20th Century Fox 's departure from Dune Entertainment upon the completion of their distribution contract at the end of 2012 . Contents ( hide ) 1
Plot 2 Cast 3 Production 3.1 Pre-production 3.2 Casting 3.3 Filming 3.4 Post-production 4 Soundtrack 5 Release 6 Reception 6.1 Box office 6.2 Home media 7
References 8 External links Plot ( edit ) In Moscow , Viktor Chagarin , a high - ranking , but corrupt , Russian official plans to incriminate former
billionaire and government whistleblower Yuri Komarov in an imminent rigged trial unless Komarov hands over a secret file believed to contain evidence
incriminating Chagarin . Separately , Jack McClane , who has been arrested after an assassination attempt , negotiates for a shorter sentence by offering to
testify against Komarov .
</item 2>

<item 3>
A Good Day to Die Hard
Komarov remarks that Jack will get to watch his father die , prompting Jack to hurl him off the rooftop into the path of the spinning helicopter rotors ,
killing him . As Jack and John reunite inside the building , Irina tries to avenge her father by ramming the helicopter , now out of ammunition , into the
building in a suicide attack . Father and son survive by leaping off the building into a large pool of water , as the helicopter crashes and explodes , killing
Irina . In the end , the McClanes return home and reunite with Lucy on the tarmac . Cast ( edit ) Bruce Willis as John McClane Jai Courtney as John `` Jack ''
McClane , Jr . Sebastian Koch as Yuri Komarov Yuliya Snigir as Irina Komarova Sergei Kolesnikov as Viktor Chagarin Radivoje Bukvić as Alik Mary Elizabeth
Winstead as Lucy McClane Additional cast members include CIA operatives : Aldis Hodge as Lt. Foxy and Cole Hauser as Mike Collins . Amaury Nolasco appears as
Murphy , an NYPD detective and McClane 's friend .
</item 3>

<item 4>
A Good Day to Die Hard
March 21 , 2013 . External links ( edit ) Official website A Good Day to Die Hard on IMDb A Good Day to Die Hard at Box Office Mojo A Good Day to Die Hard at
Rotten Tomatoes A Good Day to Die Hard at Metacritic Die Hard Films Die Hard Die Hard 2 Die Hard with a Vengeance Live Free or Die Hard A Good Day to Die Hard
Characters John McClane Video games Die Hard Die Hard Arcade Die Hard Trilogy Die Hard Trilogy 2 : Viva Las Vegas Die Hard : Nakatomi Plaza Die Hard : Vendetta
Other Die Hard : Year One The Detective novel film Nothing Lasts Forever 58 Minutes Dynamite Cop Val Verde Roderick Thorp Films directed by John Moore Behind
Enemy Lines ( 2001 ) Flight of the Phoenix ( 2004 ) The Omen ( 2006 ) Max Payne ( 2008 ) A Good Day to Die Hard ( 2013 ) I.T.
</item 4>

<item 5>
A Good Day to Die Hard
Retrieved 2012 - 02 - 22 . Jump up ^ `` Mary Elizabeth Winstead to Return as John McClane 's Daughter in A Good Day To Die Hard '' . Collider.com. 2012 - 08 -
07 . Retrieved 2012 - 08 - 07 . Jump up ^ http://www.movie-censorship.com/report.php?ID=175700 ^ Jump up to : `` A Good Day to Die Hard ' Finds a Couple of Male
and Female Villains '' . firstshowing.net. 2012 - 04 - 04 . Retrieved 2012 - 04 - 04 . Jump up ^ `` Cole Hauser Joins Bruce Willis In A Good Day To Die Hard ''
. deadline.com. 2012 - 05 - 08 . Retrieved 2012 - 05 - 08 . ^ Jump up to : Sneider , Jeff ( 2012 - 05 - 14 ) . `` Latest in ' Die Hard ' series adds two '' .
Variety . Retrieved 2012 - 05 - 31 .
</item 5>

<item 6>
A Good Day to Die Hard
A Good Day to Die Hard - wikipedia A Good Day to Die Hard Jump to : navigation , search A Good Day to Die Hard Theatrical release poster Directed by John Moore
Produced by Alex Young Wyck Godfrey Written by Skip Woods Based on Characters by Roderick Thorp Starring Bruce Willis Jai Courtney Sebastian Koch Rasha Bukvić
Cole Hauser Yuliya Snigir Music by Marco Beltrami Cinematography Jonathan Sela Edited by Dan Zimmerman Production company Giant Pictures TSG Entertainment
Distributed by 20th Century Fox Release date January 31 , 2013 ( 2013 - 01 - 31 ) ( Leicester Square ) February 14 , 2013 ( 2013 - 02 - 14 ) ( United States )
Running time 97 minutes Country United States Language English Russian Budget $92 million Box office $304.7 million A Good Day to Die Hard is a 2013 American
action thriller film and the fifth installment in the Die Hard film series . The film was directed by John Moore and written by Skip Woods , and stars Bruce
Willis as John McClane .
</item 6>

<item 7>
A Good Day to Die Hard
Casting ( edit ) Bruce Willis returned as John McClane , and has expressed a desire to shoot A Good Day to Die Hard and a sixth installment in the series before
retiring the character . When casting the role of Jack McClane , the studios considered several actors , including Liam Hemsworth and James Badge Dale , before
ultimately settling on Australian actor Jai Courtney . Mary Elizabeth Winstead also appeared in the film , reprising her role as McClane 's daughter Lucy .
However all her scenes were cut from the Extended release . Sebastian Koch played the film 's primary antagonist , Yuri Komarov , while Yuliya Snigir and Cole
Hauser featured as secondary characters Irina and Collins . The cast was completed by actors Amaury Nolasco as a friend of McClane , Pasha D. Lychnikoff as a
taxi driver , and Megalyn Echikunwoke , Anne Vyalitsyna , and Ivan Kamaras in smaller roles . Filming ( edit ) Production began in Hungary in April 2012 , with
the capital Budapest standing in for Moscow .
</item 7>

<item 8>
A Good Day to Die Hard
Rick Groen of The Globe and Mail criticized the action scenes as being `` messy '' , but concluded his review saying that the film `` continues the franchise
without undue embarrassment . '' Peter Howell of The Toronto Star remarked that Willis and Courtney made a strong estranged family duo and that the film had a
nice drinking game routine going for it with how many times McClane exclaims `` I 'm on vacation ! '' during the running time . Box Office ( edit ) A Good Day
to Die Hard grossed $67,349,198 in North America and $237,304,984 in other territories for a worldwide total of $304,654,182 , collecting more than three times
its $92 million budget . In North America , A Good Day to Die Hard grossed an estimated $840,000 from its night showings at 2,328 locations on Wednesday ,
February 13 , 2013 . The next day , at an additional 1,225 locations , the film managed to accumulate $8,239,116 , opening at # 2 behind Safe Haven .
</item 8>

<item 9>
A Good Day to Die Hard
The main plot finds McClane travelling to Russia to get his estranged son , Jack , an undercover CIA agent , out of prison . He is soon caught in the crossfire
of a global terrorist plot . Alongside Willis , the film also stars Jai Courtney , Cole Hauser , Yuliya Snigir and Sebastian Koch as the film 's villain . Talks
of a fifth Die Hard film began before the release of the fourth installment , Live Free or Die Hard , with Willis affirming that the latter would not be the
last in the series , but pre-production did not start until September 2011 , when John Moore was officially announced as the director . Filming began in April
2012 , primarily in Budapest , Hungary . A Good Day to Die Hard premiered in Los Angeles on January 31 , 2013 , coinciding with the unveiling of a Die Hard
mural at the Fox Lot , and was released in certain East and Southeast Asian countries on February 7 and in the United States and Canada on February 13 . It is
the first Die Hard film to use Dolby Atmos Surround Mixing and the first to also be released in IMAX theaters .
</item 9>

<item 10>
A Good Day to Die Hard
`` Get to the Choppa ! '' 2 : 59 23 . `` Chopper Takedown '' 3 : 26 24 . `` It 's Hard to Kill a McClane '' 2 : 59 25 . `` Triple Vodka Rhapsody '' 1 : 55 26 .
`` McClane 's Brain '' 2 : 00 Release ( edit ) # 46 TruSpeed Porsche GT3 during a race of the 2012 Supercup ( that ran as a support to the 2012 USGP ) promoting
the release date on February 14 , 2013 . On January 31 , 2013 , 20th Century Fox held a special tribute to the 25th anniversary of the Die Hard series by
unveiling a mural of a scene from Die Hard ( 1988 ) on Sound Stage 8 of the Fox Lot . Afterward , the premiere screening of A Good Day to Die Hard was shown .
Two additional premieres were held in Europe leading up to the film 's commercial release : one in Berlin , Germany on February 4 and the second in London ,
England on February 7 .
</item 10>
=== END ITEM id=cd62d14ec8976535 ===

=== ITEM id=cde28fe928622409 ===
Gold sentences and facts:
Sentence 0: Men 's ( 12 ) Basketball ( 3 ) : 1985 , 2016 , 2018 Cross Country ( 4 ) : 1966 , 1967 , 1968 , 1970 Football ( 1 ) : 2009 Indoor Track & Field ( 3 ) : 1968 , 1971 , 1979 Outdoor Track & Field ( 1 ) : 1957 Women 's ( 9 ) Cross Country ( 9 ) : 1989 , 1990 , 1991 , 1992 , 1993 , 1994 , 1998 , 2009 , 2010 see also : Big East Conference NCAA team championships List of NCAA schools with the most NCAA Division I championships
  Fact 0.0: Men's category: 12 NCAA championships in total.
  Fact 0.1: Men's basketball has 3 championships, in 1985, 2016 and 2018.
  Fact 0.2: Men's cross country has 4 championships, in 1966, 1967, 1968 and 1970.
  Fact 0.3: Men's football has 1 championship, in 2009.
  Fact 0.4: Men's indoor track and field has 3 championships, in 1968, 1971 and 1979.
  Fact 0.5: Men's outdoor track and field has 1 championship, in 1957.
  Fact 0.6: Women's category: 9 NCAA championships in total.
  Fact 0.7: Women's cross country has 9 championships, in 1989, 1990, 1991, 1992, 1993, 1994, 1998, 2009 and 2010.
  Fact 0.8: The text refers the reader to the Big East Conference NCAA team championships list.
  Fact 0.9: The text refers the reader to the list of NCAA schools with the most NCAA Division I championships.
Sentence 1: Basketball ( 3 ) : 1985 , 2016 , 2018
  Fact 1.0: Basketball has 3 championships, in 1985, 2016 and 2018.
Sentence 2: The Villanova Wildcats compete in the Big East Conference and are coached by Jay Wright . The team has traditionally divides its home schedule between its on - campus arena , the William B. Finneran Pavilion , and the Wells Fargo Center in South Philadelphia , for larger draws . During the 2017 - 18 season , the team played its entire home schedule that season at the Wells Fargo Center following the reconstruction of the Pavilion scheduled to be completed in time for the 2018 - 19 school year . Wright has taken the Wildcats to 12 postseason appearances including 7 consecutive trips ( 2005 -- 2011 ) and 5 consecutive trips ( 2013 - 2017 ) to the NCAA Tournament . The 2011 - 12 team struggled and missed postseason but Coach Wright brought a surprising young team back to the 2013 NCAA Tournament . Villanova has appeared in the NIT 17 times , winning in 1994 , and won the Big East Tournament in 1995 , 2015 and 2017 . The Wildcats have appeared in the NCAA Men 's Tournament 37 times , the 8th highest total in NCAA history have reached the Elite Eight 13 times . They have been to the Final Four of the NCAA Division I Men 's Basketball Tournament on five occasions . The 1939 team coached by Al Severance reached the inaugural NCAA Final Four played at the Palestra for the 1939 NCAA Basketball Tournament . All - American Howard Porter led a Wildcat team coached by Jack Kraft to Final Four of the 1971 NCAA University Division Basketball Tournament . Coach Rollie Massimino took Villanova to a surprising 1985 National Championship . Current Coach Jay Wright reached the Final Four at the 2009 NCAA Division I Men 's Basketball Tournament behind Dante Cunningham and Scottie Reynolds . The team most recently won the 2018 NCAA Division I Men 's Basketball Tournament for their third National Championship .
  Fact 2.0: The Villanova Wildcats compete in the Big East Conference.
  Fact 2.1: The Villanova Wildcats are coached by Jay Wright.
  Fact 2.2: The team has traditionally divided its home schedule between its on-campus arena, the William B. Finneran Pavilion, and the Wells Fargo Center in South Philadelphia.
  Fact 2.3: The Wells Fargo Center in South Philadelphia is used for larger draws.
  Fact 2.4: During the 2017-18 season, the team played its entire home schedule at the Wells Fargo Center.
  Fact 2.5: The Finneran Pavilion was being reconstructed during the 2017-18 season.
  Fact 2.6: The Finneran Pavilion reconstruction was scheduled to be completed in time for the 2018-19 school year.
  Fact 2.7: Jay Wright has taken the Wildcats to 12 postseason appearances.
  Fact 2.8: Jay Wright's postseason appearances include 7 consecutive trips to the NCAA Tournament, from 2005 to 2011.
  Fact 2.9: Jay Wright's postseason appearances include 5 consecutive trips to the NCAA Tournament, from 2013 to 2017.
  Fact 2.10: The 2011-12 Villanova team struggled and missed the postseason.
  Fact 2.11: Coach Wright brought a surprising young team back to the 2013 NCAA Tournament.
  Fact 2.12: Villanova has appeared in the NIT 17 times.
  Fact 2.13: Villanova won the NIT in 1994.
  Fact 2.14: Villanova won the Big East Tournament in 1995, 2015 and 2017.
  Fact 2.15: The Wildcats have appeared in the NCAA Men's Tournament 37 times.
  Fact 2.16: The Wildcats' 37 NCAA Men's Tournament appearances are the 8th highest total in NCAA history.
  Fact 2.17: The Wildcats have reached the Elite Eight 13 times.
  Fact 2.18: The Wildcats have been to the Final Four of the NCAA Division I Men's Basketball Tournament on five occasions.
  Fact 2.19: The 1939 Villanova team, coached by Al Severance, reached the inaugural NCAA Final Four.
  Fact 2.20: The inaugural NCAA Final Four was played at the Palestra for the 1939 NCAA Basketball Tournament.
  Fact 2.21: All-American Howard Porter led a Wildcat team, coached by Jack Kraft, to the Final Four of the 1971 NCAA University Division Basketball Tournament.
  Fact 2.22: Coach Rollie Massimino took Villanova to a surprising 1985 National Championship.
  Fact 2.23: Current coach Jay Wright reached the Final Four at the 2009 NCAA Division I Men's Basketball Tournament.
  Fact 2.24: Jay Wright's 2009 Final Four team was led by Dante Cunningham and Scottie Reynolds.
  Fact 2.25: The team most recently won the 2018 NCAA Division I Men's Basketball Tournament.
  Fact 2.26: The 2018 title was the team's third National Championship.
Sentence 3: Men 's ( 12 ) Basketball ( 3 ) : 1985 , 2016 , 2018 Cross Country ( 4 ) : 1966 , 1967 , 1968 , 1970 Football ( 1 ) : 2009 Indoor Track & Field ( 3 ) : 1968 , 1971 , 1979 Outdoor Track & Field ( 1 ) : 1957
  Fact 3.0: Men's category: 12 NCAA championships in total.
  Fact 3.1: Men's basketball has 3 championships, in 1985, 2016 and 2018.
  Fact 3.2: Men's cross country has 4 championships, in 1966, 1967, 1968 and 1970.
  Fact 3.3: Men's football has 1 championship, in 2009.
  Fact 3.4: Men's indoor track and field has 3 championships, in 1968, 1971 and 1979.
  Fact 3.5: Men's outdoor track and field has 1 championship, in 1957.

Retrieved items:
<item 1>
Villanova Wildcats
For the 2017 -- 18 season , they will play their home games at Jake Nevin Field House on the Villanova campus while the future Finneran Pavilion is undergoing
renovations . The Women 's basketball program started in 1969 and played under CIAW and AIAW through 1981 - 82 season before moving to NCAA Division I. In the
81 - 82 season , the Wildcats posted an outstanding 29 - 4 record and reached the Final Four of the 1982 AIAW Women 's Basketball Tournament before losing to
eventual National Champion , Rutgers . Through the 2009 - 10 season , the Villanova Women have a 669 - 413 alltime record since the program started in 1969 .
Previous head coaches include Liz Cawley ( 1969 -- 70 ) , Jane Sefranek - Stoltz ( 1970 -- 75 ) and Joan King ( 1975 -- 78 ) . The 1982 - 83 season was the
initial season for Villanova Women 's basketball to play NCAA Division I and they joined the Big East Conference that same season .
</item 1>

<item 2>
Villanova Wildcats
All - American Howard Porter led a Wildcat team coached by Jack Kraft to Final Four of the 1971 NCAA University Division Basketball Tournament . Coach Rollie
Massimino took Villanova to a surprising 1985 National Championship . Current Coach Jay Wright reached the Final Four at the 2009 NCAA Division I Men 's
Basketball Tournament behind Dante Cunningham and Scottie Reynolds . The team most recently won the 2018 NCAA Division I Men 's Basketball Tournament for their
third National Championship . Villanova won the 1985 NCAA Division I Men 's Basketball Tournament with a historic 66 - 64 win over top - seeded Georgetown . The
eighth - seeded Wildcats ( unranked in the final AP poll ) beat Dayton ( at Dayton ) , top - seeded Michigan , Maryland and second - seeded North Carolina to
win the Southeast Regional en route to the Final Four in Lexington , Kentucky . After defeating 2 - seed Memphis State ( now known as Memphis ) in the national
semifinals , Villanova met defending champion and ten - point - favorite Georgetown , led by Patrick Ewing , in the title game on April 1 , 1985 .
</item 2>

<item 3>
Villanova Wildcats
Coach Michael Corrado has taken the team to its only two NCAA appearances in history during his 6 - year tenure and the school is poised for greater success in
upcoming seasons . Villanova rose as high as # 7 ranking and qualified for an at - large bid in the 2011 NCAA Tournament where they lost to the University of
Denver in the first round . During the regular season , the Wildcats drew over 6500 fans to a Big East match - up with Syracuse setting a new on - campus record
for Lacrosse . Brian Karalunas became the first USILA First Team All - American in Villanova history . Through 2009 , Villanova men 's lacrosse was a member of
the Colonial Athletic Association and won the CAA tournament for their first title in 2009 . The fourth - seeded Wildcats were the lowest - seeded championship
team in conference history . That season , Villanova also made their first NCAA tournament appearance , where they were routed in the first round by first -
seeded Virginia , 18 -- 6 . The 2004 team finished 13 - 3 playing against the stiff competition of the CAA .
</item 3>

<item 4>
Villanova Wildcats
Villanova was the Big East regular season and tournament champions in both 1985 - 86 and 1986 - 87 behind All - American Shelly Pennefather . During the 2008 -
09 season , Coach Perretta led the Wildcats to the NCAA Tournament for the ninth time in his Villanova career . This marked the ninth postseason appearance in
the previous 10 years for the Wildcats . After playing in the 2000 WNIT , Villanova went to four consecutive NCAA Tournaments from 2001 to 2004 before playing
in the WNIT in 2005 , 2006 and 2008 . The 2002 - 03 season was a special one for the Wildcats . In addition to an impressive 28 - 6 overall record and a 12 - 4
Big East Conference mark , the Wildcats won their third Big East Conference Tournament championship and first since 1987 with an improbable 52 - 48 upset over
previously unbeaten Connecticut . The win snapped Connecticut 's then NCAA Division I women 's basketball record of 70 - straight victories and ended
Connecticut 's streak of nine - straight conference tournament championships .
</item 4>

<item 5>
Villanova Wildcats
NCAA team Championships ( edit ) Villanova has 21 NCAA team national championships . Men 's ( 12 ) Basketball ( 3 ) : 1985 , 2016 , 2018 Cross Country ( 4 ) :
1966 , 1967 , 1968 , 1970 Football ( 1 ) : 2009 Indoor Track & Field ( 3 ) : 1968 , 1971 , 1979 Outdoor Track & Field ( 1 ) : 1957 Women 's ( 9 ) Cross Country
( 9 ) : 1989 , 1990 , 1991 , 1992 , 1993 , 1994 , 1998 , 2009 , 2010 see also : Big East Conference NCAA team championships List of NCAA schools with the most
NCAA Division I championships Conference realignment ( edit ) In December 2012 the presidents of seven Big East schools which do not sponsor FBS football ( also
sometimes referred to as the `` Catholic 7 '' ) , including Villanova , voted unanimously to leave the Big East and form a new conference .
</item 5>

<item 6>
Villanova Wildcats
The Wildcat squad remains the lowest overall seed in tournament history to win the championship , and their overall team shooting percentage of 78.6 % remains
an NCAA tournament record for a single game . This Championship Game is often cited among the greatest upsets in college basketball history . Ed Pinckney , who
shot 5 - of - 7 and had 16 points in the game , was named the NCAA Tournament 's Most Outstanding Player . On April 4 , 2016 , the Wildcats won their second
NCAA Championship in Men 's Basketball , the first time in 31 years . They beat the North Carolina Tar Heels 77 - 74 off a buzzer - beater shot by Kris Jenkins
with 0.5 seconds remaining in regulation . Two years later , on April 2 , 2018 , Villanova earned its third national championship beating the Michigan
Wolverines 79 - 62 . Women 's Basketball ( edit ) Main article : Villanova Wildcats women 's basketball 600th win The Villanova Wildcats women 's basketball
team are coached by Harry Perretta , now serving his 34th year as head coach of the team .
</item 6>

<item 7>
Villanova Wildcats
Villanova has won 4 NCAA Men 's Cross Country Team Championships covering 1966 , 1967 , 1968 and 1970 . NCAA Individual Cross Country Champions Victor Zwolak
1963 Patrick Tiernan 2016 Swimming and Diving ( edit ) Main article : Villanova Swimming and Diving In its history , Villanova Swimming and Diving has produced
several Olympians . Most recently Maddy Crippen competed in the 400m individual medley at the 2000 Olympics . At the 2008 Olympics , Villanova was represented
by Kristina Lennox - Silva . Lennox competed for Puerto Rico and swam the 400m freestyle and 200m butterfly . NCAA team Championships ( edit ) Villanova has 21
NCAA team national championships .
</item 7>

<item 8>
Villanova Wildcats
Track and field ( edit ) Villanova University 's Track and field team has a history of athletic success that has spanned from championships in both incarnations
of the Big East Conference to NCAA championships . The men 's team has produced 69 NCAA championships , 36 indoor and 33 outdoor . The team has had 8 NCAA team
championships -- 4 cross country , 3 indoor ( ' 68 , ' 71 , ' 79 ) , 1 outdoor ( ' 57 ) . Prior to NCAA sponsoring Indoor T&F in 1965 , Villanova won 6 ICAAAA
Indoor T&F Championships ( ' 57 , ' 58 , ' 60 , ' 62 , ' 63 , ' 64 ) . Villanova has produced 28 athletes who have made appearances in the Olympics , 10 of whom
have medaled ( 7 Gold medals , 3 Silver medals ) . The men 's team has also won 112 Penn Relays Championships , which stands as the most wins by any school .
The men 's current coaches include head coach , Marcus O'Sullivan , and assistant head coach , Anthony Williams .
</item 8>

<item 9>
Villanova Wildcats
The women 's team has also had a multitude of success , producing 10 Big East team Championships and 7 NCAA team Championships . They have also produced 7
Olympians including Vicki Huber , Sonia O'Sullivan , Kim Certain , Kate Fonshell , Jen Rhines , Carmen Douma , and Carrie Tollefson . The Women 's team has won
28 Penn Relays Championships , which is the most wins by any women 's program . The current women 's coaches include head coach , Gina Procaccio , and assistant
head coach , Anthony Williams . At least one Villanova athlete has competed in every Summer Olympics since 1948 , winning a total of 13 medals ( 9 gold , 4
silver ) . Cross Country ( edit ) Villanova Men 's and Women 's Cross Country teams have a successful history of Big East and NCAA National Championships .
Villanova won both the Men 's and Women 's Big East Cross Country Team Titles in 2011 . The 2011 women 's Cross Country team finished as national runner - up
led by Sheila Reid individual title .
</item 9>

<item 10>
Villanova Wildcats
During the 2017 - 18 season , the team played its entire home schedule that season at the Wells Fargo Center following the reconstruction of the Pavilion
scheduled to be completed in time for the 2018 - 19 school year . Wright has taken the Wildcats to 12 postseason appearances including 7 consecutive trips (
2005 -- 2011 ) and 5 consecutive trips ( 2013 - 2017 ) to the NCAA Tournament . The 2011 - 12 team struggled and missed postseason but Coach Wright brought a
surprising young team back to the 2013 NCAA Tournament . Villanova has appeared in the NIT 17 times , winning in 1994 , and won the Big East Tournament in 1995
, 2015 and 2017 . The Wildcats have appeared in the NCAA Men 's Tournament 37 times , the 8th highest total in NCAA history have reached the Elite Eight 13
times . They have been to the Final Four of the NCAA Division I Men 's Basketball Tournament on five occasions . The 1939 team coached by Al Severance reached
the inaugural NCAA Final Four played at the Palestra for the 1939 NCAA Basketball Tournament .
</item 10>
=== END ITEM id=cde28fe928622409 ===

=== ITEM id=0d7f68455bc8acb2 ===
Gold sentences and facts:
Sentence 0: `` A Whiter Shade of Pale '' is the debut single by the British rock band Procol Harum , released 12 May 1967 . The single reached number one in the UK Singles Chart on 8 June 1967 and stayed there for six weeks . Without much promotion , it reached number 5 on the Bilboard pop chart in the United States . One of the anthems of the 1967 Summer of Love , it is one of fewer than 30 singles to have sold over 10 million copies worldwide .
  Fact 0.0: "A Whiter Shade of Pale" is the debut single by the British rock band Procol Harum.
  Fact 0.1: "A Whiter Shade of Pale" was released on 12 May 1967.
  Fact 0.2: "A Whiter Shade of Pale" reached number one in the UK Singles Chart on 8 June 1967.
  Fact 0.3: "A Whiter Shade of Pale" stayed at number one in the UK Singles Chart for six weeks.
  Fact 0.4: Without much promotion, "A Whiter Shade of Pale" reached number 5 on the Billboard pop chart in the United States.
  Fact 0.5: "A Whiter Shade of Pale" is one of the anthems of the 1967 Summer of Love.
  Fact 0.6: "A Whiter Shade of Pale" is one of fewer than 30 singles to have sold over 10 million copies worldwide.

Retrieved items:
<item 1>
A Whiter Shade of Pale
In his 1981 article on the musical and societal developments of 1967 , for The History of Rock , sociomusicologist Simon Frith described `` A Whiter Shade of
Pale '' as the year 's `` most distinctive single '' , through its combination of `` white soul vocal and a Bach organ exercise '' and enigmatic lyrics that ``
hinted at a vital secret open only to people in the right , drug - determined , state of mind '' . In the United States , the single reached number 5 on the
Billboard Hot 100 and sold over 1 million copies . It also peaked at number 22 on the soul charts there . In the Netherlands , the song entered the chart at
number 1 in June 1967 and again reached number 1 in July 1972 . A May 1972 re-release on Fly Records stayed in the UK charts for a total of 12 weeks and peaked
at number 13 . `` A Whiter Shade of Pale '' has continued to receive critical acclaim .
</item 1>

<item 2>
A Whiter Shade of Pale
Lyrics ( edit ) Reid got the title and starting point for the song at a party . He overheard someone at the party saying to a woman , `` You 've turned a whiter
shade of pale '' , and the phrase stuck in his mind . The original lyrics had four verses , of which only two are heard on the original recording . The third
verse has been heard in live performances by Procol Harum , and more seldom also the fourth . The author of Procol Harum : Beyond the Pale , Claes Johansen ,
suggests that the song `` deals in metaphorical form with a male / female relationship which after some negotiation ends in a sexual act '' . This is supported
by Tim de Lisle in Lives of the Great Songs , who remarks that the lyrics concern a drunken seduction , which is described through references to sex as a form
of travel , usually nautical , using mythical and literary journeys . Other observers have also commented that the lyrics concern a sexual relationship .
</item 2>

<item 3>
A Whiter Shade of Pale
In two weeks it reached number 1 , where it stayed for six weeks . Writing in 2005 , Jim Irvin of Mojo said that its arrival at number 1 on 8 June 1967 , on the
same day that the Beatles ' Sgt . Pepper 's Lonely Hearts Club Band topped the national albums chart , marked the start of the Summer of Love in Britain .
According to music journalist and author Harvey Kubernik , in the context of the Summer of Love , `` A Whiter Shade of Pale '' was the `` one song ( that )
stood above all others , its Everest - like status conferred by no less than John Lennon and Paul McCartney , who were enthralled by the Chaucerian wordplay and
heavenly Baroque accompaniment '' . Kubernik also writes that , amid the search for higher consciousness during the flower power era , the song `` galvanised a
congregation of disaffected youth dismissive of traditional religion but anxious to achieve spiritual salvation '' .
</item 3>

<item 4>
A Whiter Shade of Pale
ISBN 978 - 1 - 4549 - 2052 - 6 . External links ( edit ) Lyrics of this song at MetroLyrics A Whiter Shade of Pale : authorised lyrics of all four verses ``
Procol Harum : The ultimate tale of A Whiter Shade Of Pale '' , article on the song 's 40th anniversary , from Classic Rock magazine Bill Eyden Obituary Procol
Harum -- A Whiter Shade Of Pale ,
</item 4>

<item 5>
A Whiter Shade of Pale
`` A Whiter Shade of Pale '' has continued to receive critical acclaim . Along with Queen 's `` Bohemian Rhapsody '' , `` A Whiter Shade of Pale '' was jointly
recognised as `` The Best British Pop Single 1952 -- 1977 '' at the BRIT Awards , part of Elizabeth II 's Silver Jubilee . In 1998 the song was inducted into
the Grammy Hall of Fame . In 2004 , it appeared at number 57 on Rolling Stone magazine 's list of the `` 500 Greatest Songs of All Time '' . British TV station
Channel 4 placed the song at number 19 in its chart of `` The 100 Greatest No. 1 Singles '' . In 2018 , the song was inducted into the Rock and Roll Hall of
Fame in a new category for singles . Promotional films ( edit ) One of the clips for the song was filmed at Witley Court , part of which was gutted by fire in
1937 . The first promotional clip for `` A Whiter Shade of Pale '' was shot in the ruins of Witley Court in Worcestershire , England .
</item 5>

<item 6>
A Whiter Shade of Pale
One of the anthems of the 1967 Summer of Love , it is one of fewer than 30 singles to have sold over 10 million copies worldwide . With its Bach - derived
instrumental melody , soulful vocals , and unusual lyrics -- by the song 's co-authors Gary Brooker , Keith Reid and Matthew Fisher -- `` A Whiter Shade of Pale
'' reached No. 1 in several countries when released in 1967 . In the years since , it has become an enduring classic . It was the most played song in the last
75 years in public places in the UK ( as of 2009 ) , and the United Kingdom performing rights group Phonographic Performance Limited in 2004 recognised it as
the most - played record by British broadcasting of the past 70 years . Also in 2004 , Rolling Stone placed `` A Whiter Shade of Pale '' 57th on its list of the
500 Greatest Songs of All Time . In 1977 , the song was named joint winner ( along with Queen 's `` Bohemian Rhapsody '' ) of `` The Best British Pop Single
1952 -- 1977 '' at the Brit Awards .
</item 6>

<item 7>
A Whiter Shade of Pale
A Whiter Shade of Pale - wikipedia A Whiter Shade of Pale Jump to : navigation , search `` A Whiter Shade of Pale '' A-side label of the UK vinyl single Single
by Procol Harum from the album Procol Harum B - side `` Lime Street Blues '' Released 12 May 1967 Format 7 - inch record Recorded Olympic Sound Studios , London
Genre Baroque pop Length 4 : 03 Label Deram Songwriter ( s ) Gary Brooker , Keith Reid , Matthew Fisher Producer ( s ) Denny Cordell Procol Harum singles
chronology `` A Whiter Shade of Pale '' ( 1967 ) `` Homburg '' ( 1967 ) `` A Whiter Shade of Pale '' ( 1967 ) `` Homburg '' ( 1967 ) `` A Whiter Shade of Pale
'' is the debut single by the British rock band Procol Harum , released 12 May 1967 . The single reached number one in the UK Singles Chart on 8 June 1967 and
stayed there for six weeks . Without much promotion , it reached number 5 on the Bilboard pop chart in the United States .
</item 7>

<item 8>
A Whiter Shade of Pale
live video from Eagle Rock Entertainment 's official channel on YouTube `` A Whiter Shade of Pale '' Scopitone film on YouTube Preceded by `` Silence is Golden
'' by The Tremeloes UK Singles Chart number one single 8 June 1967 -- 18 July 1967 ( six weeks ) Succeeded by `` All You Need Is Love '' by The Beatles Preceded
by `` Black Velvet Band '' by Johnny McEvoy Ireland 1967 number one single 22 June 1967 -- 19 July 1967 ( five weeks ) Succeeded by `` Black Velvet Band '' by
Johnny McEvoy Preceded by `` This is My Song '' by Petula Clark Australia 1967 number one single 8 July 1967 -- 28 July 1967 ( three weeks ) Succeeded by `` All
You Need is Love / Baby ,
</item 8>

<item 9>
A Whiter Shade of Pale
Its unusually allusive and referential lyrics are much more complex than most lyrics of the time ( for example , the chorus focuses on Chaucer 's `` The Miller
's Tale '' ) . Thus , this piece can be considered an early example of progressive rock . The phrase a whiter shade of pale has since gained widespread use in
the English language , noticed by several dictionaries . As such , the phrase is today often used in contexts independent of any consideration of the song . It
has also been heavily paraphrased , in forms like `` an Xer shade of Y '' , to the extent that it has been recognised as a snowclone -- a type of cliché and
phrasal template . Composition ( edit ) The song 's distinctive , Bach - inspired organ part was played on a Hammond M102 . The song is in moderate time in C
major and is characterised by the bassline moving stepwise downwards in a repeated pattern throughout . In classical music this is known as a ground bass . The
harmonic structure is identical for the organ melody , the verse and the chorus , except that the chorus finishes with a cadence .
</item 9>

<item 10>
A Whiter Shade of Pale
Contents ( hide ) 1 Recording and personnel 2 Lyrics 3 Composition 4 Release and reception 5 Promotional films 6 Authorship lawsuit 7 Chart performance 7.1
Weekly singles charts 7.2 Year - end charts 8 Annie Lennox version 8.1 Charts 9 References 10 Sources 11 External links Recording and personnel ( edit ) The
song was performed and recorded at Olympic Sound Studios in London , England , with Gary Brooker providing the vocals and piano , Matthew Fisher on a Hammond M
- 102 organ , David Knights on bass and Ray Royer on guitar . Drums were by session drummer Bill Eyden . A few days later , the song was re-recorded with the
band 's then newly recruited drummer Bobby Harrison , but that version was discarded , and one of the original mono recordings was chosen for release . Producer
for the record was Denny Cordell , and Keith Grant was the sound engineer . The song was included on the US release of the Procol Harum album , in September
1967 , but not on the later UK version . Lyrics ( edit ) Reid got the title and starting point for the song at a party .
</item 10>
=== END ITEM id=0d7f68455bc8acb2 ===

=== ITEM id=7f39b4fe3eb0f2b8 ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) `` Pilot '' Adam Davidson Robert Kirkman & Dave Erickson August 23 , 2015 ( 2015 - 08 - 23 ) 10.13 Nick awakens in a heroin den in an abandoned church , to find his girlfriend Gloria eating a corpse . While fleeing , he is hit by a car and hospitalized . The doctor tells Madison and Travis that Nick 's claims about the incident are heroin hallucinations , but Travis believes Nick after visiting the church himself . Alicia becomes more worried about Nick 's chemical dependency . The next day , school closes early due to the high levels of absenteeism and rumors of an epidemic . Nick escapes from the hospital and meets with Calvin , hoping to learn if the drugs Calvin sold him caused him to hallucinate in the church . Calvin tries to kill Nick to prevent him from exposing Calvin as a drug dealer or impugning the quality of his drugs . In the ensuing struggle , Calvin is mortally shot . After Travis and Madison arrive , the zombified Calvin attacks them . Nick runs over Calvin repeatedly with Travis ' truck , and the three watch in disbelief as the mutilated Calvin is still able to turn his head towards them . `` So Close , Yet So Far '' Adam Davidson Marco Ramirez August 30 , 2015 ( 2015 - 08 - 30 ) 8.18 After the run - in with zombified Calvin , Nick , Madison , and Travis choose to flee to the desert , and they want Alicia , Liza , and Chris to follow . Alicia finds Matt ill in his disheveled house . Travis arrives and sees a bite on Matt , who convinces Alicia to leave . The group returns to Madison 's home to gather supplies . Nick suffers from heroin withdrawal , so Madison drives to her school to get him Oxycodone . There , she encounters Tobias scavenging food . A zombified Artie tries to bite Tobias , so Madison kills Artie and drives Tobias home . Chris ' bus is trapped in a traffic jam caused by a zombie shooting by the police . He films the event and joins in a protest against that and other recent fatal police shootings , when Travis and Liza meet up with him . A riot erupts after police shoot down another zombie , but the three Manawas find refuge with the Salazars in their gated barbershop - home . Travis tells Madison to take the kids to the desert without him ; he will catch up . Alicia witnesses her zombified neighbor , Mr. Dawson , attacking Mrs. Cruz across the street , but Madison prevents her from intervening . The group inside the barbershop remains trapped , while the riot outside intensifies . `` The Dog '' Adam Davidson Jack LoGiudice September 13 , 2015 ( 2015 - 09 - 13 ) 7.19 While a riot rages outside , a mob sets fire to the store adjoining the barbershop , forcing the Salazars and Manawas to flee . The group reaches Travis ' truck and escapes , but not before Griselda is injured by a collapsing scaffold . Unable to reach a hospital , the group drives to Madison 's house , where Nick , Madison , and Alicia temporarily flee when the zombified Mr. Dawson attempts to enter , attracted by the barking dog Nick had let in . Nick leads Madison and Alicia to the Trans ' house next door , where they take a shotgun . Travis arrives and is attacked by Mr. Dawson , who is shot and killed by Daniel . All three families decide to stay the night and evacuate in the morning . Nurse Liza tends to Griselda 's injured foot but notes that Griselda will die if not treated by a doctor . Ofelia tells Daniel they should flee with Travis , but Daniel insists his family can survive alone and will join his cousin later . The next morning , as the Clarks and Manawas start driving away , the National Guard arrives and quarantines the block . While Travis says , `` It 's going to get better , '' Daniel laments that it 's , `` too late , '' as he watches a guardsman mark the neighboring house . `` Not Fade Away '' Kari Skogland Meaghan Oppenheimer September 20 , 2015 ( 2015 - 09 - 20 ) 6.62 Days after the National Guard quarantines the neighborhood into a Safe Zone , residents try to live normally . Tensions build under the military rule . Madison is stressed by extra work caused by her home 's being overcrowded and Travis ' new role as civilian liaison . Chris shows a video to Travis and Madison of a light signaling from the Dead Zone . Travis talks Doug into getting mental help . Liza medically helps neighbors . Nick steals morphine from Hector via IV drip . Ofelia kisses Adams , who was unable to get Griselda 's medicine . Madison exchanges signals with the flashing light in the Dead Zone . She sneaks outside the fence to investigate and finds evidence that the guardsmen killed civilians , even the uninfected . Travis learns that Doug has been hospitalized for his mental issues . Dr. Exner determines that Liza is not technically a nurse . Daniel warns Madison of his El Salvador experience , when the sick were taken under the guise of receiving hospitalization but instead killed . Soldiers take Griselda and Nick to a hospital , but Nick 's family protests his departure . Liza agrees to go to assist the medical team , despite not wanting to leave her son . Travis retreats to the roof and sees the signal from the Dead Zone . Seconds later , he sees and hears gunfire , followed by darkness . 5 5 `` Cobalt '' Kari Skogland David Wiener September 27 , 2015 ( 2015 - 09 - 27 ) 6.66 In a military cell , Strand bribes a guardsman to save feverish Nick from being moved . Liza helps Dr. Exner with patients at the hospital . Chris is devastated that Liza left voluntarily to help at the hospital , but Travis promises to bring her back . Madison discovers Daniel detaining Adams in the Trans ' basement . Alicia and Chris get drunk in and vandalize the abandoned home of a wealthy family . Strand recruits Nick for an escape plot . Travis convinces Moyers ' squad to take him to the hospital to check on his friends . While en route , Moyers encourages Travis to shoot a zombie , but Travis is emotionally unable to pull the trigger . The soldiers stop to assist another squad in a building infested by zombies , and most of those soldiers , including Moyers , are overcome . The few survivors flee and drop off Travis near the Safe Zone . Travis learns that Daniel tortured Adams into revealing what `` Cobalt '' means : in the morning , all civilians will be killed , and the guardsmen will evacuate the city . Griselda dies of septic shock at the hospital ; Liza shoots her brain to prevent reanimation . Daniel visits a nearby sports arena to verify Adams ' story that it was sealed with 2,000 now - zombified civilians inside . 6 6 `` The Good Man '' Stefan Schwartz Robert Kirkman & Dave Erickson October 4 , 2015 ( 2015 - 10 - 04 ) 6.86 The group drives to the National Guard 's headquarters to rescue Liza , Griselda , and Nick . Adams agrees to be their guide when let go by Travis . The group infiltrates the base after Daniel distracts the guards by leading a horde of walkers from the arena . Travis , Madison , Daniel , and Ofelia go inside , while Alicia and Chris stay behind . Meanwhile , the walkers breach the perimeter defenses and swarm the base . Travis ' group reach the holding cells and set the detainees free before reuniting with Nick , Liza , and Strand . They try to escape through the medical ward , where they discover Dr. Exner has euthanized all of the patients . Dr. Exner tells them of an escape route before presumably committing suicide . Before they can escape , the group encounters Adams , who shoots Ofelia in the arm . Enraged , Travis brutally beats Adams and leaves him for dead . Strand leads the group to his oceanside mansion , where he reveals to Nick that he owns a yacht which he plans to escape on , called the Abigail . On the beach , Liza reveals to Madison that she had been bitten during the escape . Liza pleads with Madison and Travis to euthanize her before she turns . Travis promises to protect Chris before shooting Liza .
  Fact 0.0: The table lists each episode's overall number, season number, title, director, writer, original air date, and U.S. viewers in millions.
  Fact 0.1: Episode 'Pilot' was directed by Adam Davidson and written by Robert Kirkman and Dave Erickson.
  Fact 0.2: Episode 'Pilot' originally aired on August 23, 2015.
  Fact 0.3: Episode 'Pilot' had 10.13 million U.S. viewers.
  Fact 0.4: Nick awakens in a heroin den in an abandoned church, to find his girlfriend Gloria eating a corpse.
  Fact 0.5: While fleeing, Nick is hit by a car and hospitalized.
  Fact 0.6: The doctor tells Madison and Travis that Nick's claims about the incident are heroin hallucinations.
  Fact 0.7: Travis believes Nick after Travis visits the church himself.
  Fact 0.8: Alicia becomes more worried about Nick's chemical dependency.
  Fact 0.9: The next day, school closes early due to high levels of absenteeism and rumors of an epidemic.
  Fact 0.10: Nick escapes from the hospital and meets with Calvin, hoping to learn if the drugs Calvin sold him caused him to hallucinate in the church.
  Fact 0.11: Calvin tries to kill Nick to prevent him from exposing Calvin as a drug dealer or impugning the quality of his drugs.
  Fact 0.12: In the ensuing struggle, Calvin is mortally shot.
  Fact 0.13: After Travis and Madison arrive, the zombified Calvin attacks them.
  Fact 0.14: Nick runs over Calvin repeatedly with Travis's truck.
  Fact 0.15: The three watch in disbelief as the mutilated Calvin is still able to turn his head towards them.
  Fact 0.16: Episode 'So Close, Yet So Far' was directed by Adam Davidson and written by Marco Ramirez.
  Fact 0.17: Episode 'So Close, Yet So Far' originally aired on August 30, 2015.
  Fact 0.18: Episode 'So Close, Yet So Far' had 8.18 million U.S. viewers.
  Fact 0.19: After the run-in with zombified Calvin, Nick, Madison, and Travis choose to flee to the desert.
  Fact 0.20: Nick, Madison, and Travis want Alicia, Liza, and Chris to follow them to the desert.
  Fact 0.21: Alicia finds Matt ill in his disheveled house.
  Fact 0.22: Travis arrives and sees a bite on Matt.
  Fact 0.23: Matt convinces Alicia to leave.
  Fact 0.24: The group returns to Madison's home to gather supplies.
  Fact 0.25: Nick suffers from heroin withdrawal, so Madison drives to her school to get him Oxycodone.
  Fact 0.26: At the school, Madison encounters Tobias scavenging food.
  Fact 0.27: A zombified Artie tries to bite Tobias, so Madison kills Artie and drives Tobias home.
  Fact 0.28: Chris's bus is trapped in a traffic jam caused by a zombie shooting by the police.
  Fact 0.29: Chris films the event and joins in a protest against that and other recent fatal police shootings, when Travis and Liza meet up with him.
  Fact 0.30: A riot erupts after police shoot down another zombie.
  Fact 0.31: The three Manawas find refuge with the Salazars in their gated barbershop-home.
  Fact 0.32: Travis tells Madison to take the kids to the desert without him, and he will catch up.
  Fact 0.33: Alicia witnesses her zombified neighbor, Mr. Dawson, attacking Mrs. Cruz across the street.
  Fact 0.34: Madison prevents Alicia from intervening.
  Fact 0.35: The group inside the barbershop remains trapped, while the riot outside intensifies.
  Fact 0.36: Episode 'The Dog' was directed by Adam Davidson and written by Jack LoGiudice.
  Fact 0.37: Episode 'The Dog' originally aired on September 13, 2015.
  Fact 0.38: Episode 'The Dog' had 7.19 million U.S. viewers.
  Fact 0.39: While a riot rages outside, a mob sets fire to the store adjoining the barbershop, forcing the Salazars and Manawas to flee.
  Fact 0.40: The group reaches Travis's truck and escapes, but not before Griselda is injured by a collapsing scaffold.
  Fact 0.41: Unable to reach a hospital, the group drives to Madison's house.
  Fact 0.42: Nick, Madison, and Alicia temporarily flee when the zombified Mr. Dawson attempts to enter.
  Fact 0.43: Mr. Dawson is attracted by the barking dog Nick had let in.
  Fact 0.44: Nick leads Madison and Alicia to the Trans' house next door, where they take a shotgun.
  Fact 0.45: Travis arrives and is attacked by Mr. Dawson.
  Fact 0.46: Mr. Dawson is shot and killed by Daniel.
  Fact 0.47: All three families decide to stay the night and evacuate in the morning.
  Fact 0.48: Nurse Liza tends to Griselda's injured foot but notes that Griselda will die if not treated by a doctor.
  Fact 0.49: Ofelia tells Daniel they should flee with Travis.
  Fact 0.50: Daniel insists his family can survive alone and will join his cousin later.
  Fact 0.51: The next morning, as the Clarks and Manawas start driving away, the National Guard arrives and quarantines the block.
  Fact 0.52: Travis says 'It's going to get better.'
  Fact 0.53: Daniel laments that it's 'too late' as he watches a guardsman mark the neighboring house.
  Fact 0.54: Episode 'Not Fade Away' was directed by Kari Skogland and written by Meaghan Oppenheimer.
  Fact 0.55: Episode 'Not Fade Away' originally aired on September 20, 2015.
  Fact 0.56: Episode 'Not Fade Away' had 6.62 million U.S. viewers.
  Fact 0.57: Days after the National Guard quarantines the neighborhood into a Safe Zone, residents try to live normally.
  Fact 0.58: Tensions build under the military rule.
  Fact 0.59: Madison is stressed by extra work caused by her home's being overcrowded.
  Fact 0.60: Madison is stressed by Travis's new role as civilian liaison.
  Fact 0.61: Chris shows a video to Travis and Madison of a light signaling from the Dead Zone.
  Fact 0.62: Travis talks Doug into getting mental help.
  Fact 0.63: Liza medically helps neighbors.
  Fact 0.64: Nick steals morphine from Hector via IV drip.
  Fact 0.65: Ofelia kisses Adams, who was unable to get Griselda's medicine.
  Fact 0.66: Madison exchanges signals with the flashing light in the Dead Zone.
  Fact 0.67: Madison sneaks outside the fence to investigate.
  Fact 0.68: Madison finds evidence that the guardsmen killed civilians, even the uninfected.
  Fact 0.69: Travis learns that Doug has been hospitalized for his mental issues.
  Fact 0.70: Dr. Exner determines that Liza is not technically a nurse.
  Fact 0.71: Daniel warns Madison of his El Salvador experience, when the sick were taken under the guise of receiving hospitalization but instead killed.
  Fact 0.72: Soldiers take Griselda and Nick to a hospital, but Nick's family protests his departure.
  Fact 0.73: Liza agrees to go to assist the medical team, despite not wanting to leave her son.
  Fact 0.74: Travis retreats to the roof and sees the signal from the Dead Zone.
  Fact 0.75: Seconds later, Travis sees and hears gunfire, followed by darkness.
  Fact 0.76: Episode 'Cobalt' is overall number 5 and season number 5.
  Fact 0.77: Episode 'Cobalt' was directed by Kari Skogland and written by David Wiener.
  Fact 0.78: Episode 'Cobalt' originally aired on September 27, 2015.
  Fact 0.79: Episode 'Cobalt' had 6.66 million U.S. viewers.
  Fact 0.80: In a military cell, Strand bribes a guardsman to save feverish Nick from being moved.
  Fact 0.81: Liza helps Dr. Exner with patients at the hospital.
  Fact 0.82: Chris is devastated that Liza left voluntarily to help at the hospital.
  Fact 0.83: Travis promises to bring Liza back.
  Fact 0.84: Madison discovers Daniel detaining Adams in the Trans' basement.
  Fact 0.85: Alicia and Chris get drunk in and vandalize the abandoned home of a wealthy family.
  Fact 0.86: Strand recruits Nick for an escape plot.
  Fact 0.87: Travis convinces Moyers' squad to take him to the hospital to check on his friends.
  Fact 0.88: While en route, Moyers encourages Travis to shoot a zombie.
  Fact 0.89: Travis is emotionally unable to pull the trigger.
  Fact 0.90: The soldiers stop to assist another squad in a building infested by zombies.
  Fact 0.91: Most of those soldiers, including Moyers, are overcome.
  Fact 0.92: The few survivors flee and drop off Travis near the Safe Zone.
  Fact 0.93: Travis learns that Daniel tortured Adams into revealing what 'Cobalt' means.
  Fact 0.94: 'Cobalt' means that in the morning, all civilians will be killed, and the guardsmen will evacuate the city.
  Fact 0.95: Griselda dies of septic shock at the hospital.
  Fact 0.96: Liza shoots Griselda's brain to prevent reanimation.
  Fact 0.97: Daniel visits a nearby sports arena to verify Adams' story that it was sealed with 2,000 now-zombified civilians inside.
  Fact 0.98: Episode 'The Good Man' is overall number 6 and season number 6.
  Fact 0.99: Episode 'The Good Man' was directed by Stefan Schwartz and written by Robert Kirkman and Dave Erickson.
  Fact 0.100: Episode 'The Good Man' originally aired on October 4, 2015.
  Fact 0.101: Episode 'The Good Man' had 6.86 million U.S. viewers.
  Fact 0.102: The group drives to the National Guard's headquarters to rescue Liza, Griselda, and Nick.
  Fact 0.103: Adams agrees to be their guide when let go by Travis.
  Fact 0.104: The group infiltrates the base after Daniel distracts the guards by leading a horde of walkers from the arena.
  Fact 0.105: Travis, Madison, Daniel, and Ofelia go inside, while Alicia and Chris stay behind.
  Fact 0.106: Meanwhile, the walkers breach the perimeter defenses and swarm the base.
  Fact 0.107: Travis's group reaches the holding cells and sets the detainees free.
  Fact 0.108: Travis's group reunites with Nick, Liza, and Strand before trying to escape.
  Fact 0.109: The group tries to escape through the medical ward.
  Fact 0.110: The group discovers Dr. Exner has euthanized all of the patients in the medical ward.
  Fact 0.111: Dr. Exner tells them of an escape route before presumably committing suicide.
  Fact 0.112: Before they can escape, the group encounters Adams, who shoots Ofelia in the arm.
  Fact 0.113: Enraged, Travis brutally beats Adams and leaves him for dead.
  Fact 0.114: Strand leads the group to his oceanside mansion.
  Fact 0.115: At the mansion, Strand reveals to Nick that he owns a yacht which he plans to escape on, called the Abigail.
  Fact 0.116: On the beach, Liza reveals to Madison that she had been bitten during the escape.
  Fact 0.117: Liza pleads with Madison and Travis to euthanize her before she turns.
  Fact 0.118: Travis promises to protect Chris before shooting Liza.

Retrieved items:
<item 1>
Fear the Walking Dead (season 1)
overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) `` Pilot '' Adam Davidson Robert Kirkman & Dave Erickson August
23 , 2015 ( 2015 - 08 - 23 ) 10.13 Nick awakens in a heroin den in an abandoned church , to find his girlfriend Gloria eating a corpse . While fleeing , he is
hit by a car and hospitalized . The doctor tells Madison and Travis that Nick 's claims about the incident are heroin hallucinations , but Travis believes Nick
after visiting the church himself . Alicia becomes more worried about Nick 's chemical dependency . The next day , school closes early due to the high levels of
absenteeism and rumors of an epidemic . Nick escapes from the hospital and meets with Calvin , hoping to learn if the drugs Calvin sold him caused him to
hallucinate in the church . Calvin tries to kill Nick to prevent him from exposing Calvin as a drug dealer or impugning the quality of his drugs . In the
ensuing struggle , Calvin is mortally shot . After Travis and Madison arrive , the zombified Calvin attacks them .
</item 1>

<item 2>
Fear the Walking Dead (season 1)
Dr. Exner determines that Liza is not technically a nurse . Daniel warns Madison of his El Salvador experience , when the sick were taken under the guise of
receiving hospitalization but instead killed . Soldiers take Griselda and Nick to a hospital , but Nick 's family protests his departure . Liza agrees to go to
assist the medical team , despite not wanting to leave her son . Travis retreats to the roof and sees the signal from the Dead Zone . Seconds later , he sees
and hears gunfire , followed by darkness . 5 5 `` Cobalt '' Kari Skogland David Wiener September 27 , 2015 ( 2015 - 09 - 27 ) 6.66 In a military cell , Strand
bribes a guardsman to save feverish Nick from being moved . Liza helps Dr. Exner with patients at the hospital . Chris is devastated that Liza left voluntarily
to help at the hospital , but Travis promises to bring her back . Madison discovers Daniel detaining Adams in the Trans ' basement . Alicia and Chris get drunk
in and vandalize the abandoned home of a wealthy family .
</item 2>

<item 3>
Fear the Walking Dead (season 1)
6 6 `` The Good Man '' Stefan Schwartz Robert Kirkman & Dave Erickson October 4 , 2015 ( 2015 - 10 - 04 ) 6.86 The group drives to the National Guard 's
headquarters to rescue Liza , Griselda , and Nick . Adams agrees to be their guide when let go by Travis . The group infiltrates the base after Daniel distracts
the guards by leading a horde of walkers from the arena . Travis , Madison , Daniel , and Ofelia go inside , while Alicia and Chris stay behind . Meanwhile ,
the walkers breach the perimeter defenses and swarm the base . Travis ' group reach the holding cells and set the detainees free before reuniting with Nick ,
Liza , and Strand . They try to escape through the medical ward , where they discover Dr. Exner has euthanized all of the patients . Dr. Exner tells them of an
escape route before presumably committing suicide . Before they can escape , the group encounters Adams , who shoots Ofelia in the arm . Enraged , Travis
brutally beats Adams and leaves him for dead .
</item 3>

<item 4>
Fear the Walking Dead (season 1)
Unable to reach a hospital , the group drives to Madison 's house , where Nick , Madison , and Alicia temporarily flee when the zombified Mr. Dawson attempts to
enter , attracted by the barking dog Nick had let in . Nick leads Madison and Alicia to the Trans ' house next door , where they take a shotgun . Travis arrives
and is attacked by Mr. Dawson , who is shot and killed by Daniel . All three families decide to stay the night and evacuate in the morning . Nurse Liza tends to
Griselda 's injured foot but notes that Griselda will die if not treated by a doctor . Ofelia tells Daniel they should flee with Travis , but Daniel insists his
family can survive alone and will join his cousin later . The next morning , as the Clarks and Manawas start driving away , the National Guard arrives and
quarantines the block . While Travis says , `` It 's going to get better , '' Daniel laments that it 's , `` too late , '' as he watches a guardsman mark the
neighboring house .
</item 4>

<item 5>
Fear the Walking Dead (season 1)
Chris ' bus is trapped in a traffic jam caused by a zombie shooting by the police . He films the event and joins in a protest against that and other recent
fatal police shootings , when Travis and Liza meet up with him . A riot erupts after police shoot down another zombie , but the three Manawas find refuge with
the Salazars in their gated barbershop - home . Travis tells Madison to take the kids to the desert without him ; he will catch up . Alicia witnesses her
zombified neighbor , Mr. Dawson , attacking Mrs. Cruz across the street , but Madison prevents her from intervening . The group inside the barbershop remains
trapped , while the riot outside intensifies . `` The Dog '' Adam Davidson Jack LoGiudice September 13 , 2015 ( 2015 - 09 - 13 ) 7.19 While a riot rages outside
, a mob sets fire to the store adjoining the barbershop , forcing the Salazars and Manawas to flee . The group reaches Travis ' truck and escapes , but not
before Griselda is injured by a collapsing scaffold .
</item 5>

<item 6>
Fear the Walking Dead (season 1)
`` Not Fade Away '' Kari Skogland Meaghan Oppenheimer September 20 , 2015 ( 2015 - 09 - 20 ) 6.62 Days after the National Guard quarantines the neighborhood
into a Safe Zone , residents try to live normally . Tensions build under the military rule . Madison is stressed by extra work caused by her home 's being
overcrowded and Travis ' new role as civilian liaison . Chris shows a video to Travis and Madison of a light signaling from the Dead Zone . Travis talks Doug
into getting mental help . Liza medically helps neighbors . Nick steals morphine from Hector via IV drip . Ofelia kisses Adams , who was unable to get Griselda
's medicine . Madison exchanges signals with the flashing light in the Dead Zone . She sneaks outside the fence to investigate and finds evidence that the
guardsmen killed civilians , even the uninfected . Travis learns that Doug has been hospitalized for his mental issues . Dr. Exner determines that Liza is not
technically a nurse .
</item 6>

<item 7>
Fear the Walking Dead (season 1)
1 Talking Dead Fear the Walking Dead Episodes season 1 Web series The Walking Dead Torn Apart Cold Storage Fear the Walking Dead Flight 462 Miscellaneous Video
games Season One `` 400 Days '' Two Michonne Three : A New Frontier Other games Survival Instinct Road to Survival Rides The Walking Dead : The Ride Novels Rise
of the Governor The Road to Woodbury The Fall of the Governor Descent Invasion Search and Destroy Return to Woodbury Characters Lists Comic book series
Television series Aaron Alden Amy Jessie Anderson Andrea Axel Tara Chambler Clementine Daryl Dixon Merle Dixon Dwight Rosita Espinosa Lee Everett King Ezekiel
Abraham Ford Gareth The Governor Gregory Beth Greene Hershel Greene Maggie Greene Carl Grimes Lori Grimes Rick Grimes Dale Horvath Jadis Morgan Jones Lilly
Milton Mamet Caesar Martinez Deanna Monroe Paul `` Jesus '' Monroe Spencer Monroe Morales Michonne Negan Olivia The Prisoners Carol Peletier Sophia Peletier
Eugene Porter Glenn Rhee Lizzie and Mika Samuels Simon Gabriel Stokes Bob Stookey T - Dog Shane Walsh Sasha Williams Tyreese Williams Retrieved from ``
https://en.wikipedia.org/w/index.
</item 7>

<item 8>
Fear the Walking Dead (season 1)
Alicia and Chris get drunk in and vandalize the abandoned home of a wealthy family . Strand recruits Nick for an escape plot . Travis convinces Moyers ' squad
to take him to the hospital to check on his friends . While en route , Moyers encourages Travis to shoot a zombie , but Travis is emotionally unable to pull the
trigger . The soldiers stop to assist another squad in a building infested by zombies , and most of those soldiers , including Moyers , are overcome . The few
survivors flee and drop off Travis near the Safe Zone . Travis learns that Daniel tortured Adams into revealing what `` Cobalt '' means : in the morning , all
civilians will be killed , and the guardsmen will evacuate the city . Griselda dies of septic shock at the hospital ; Liza shoots her brain to prevent
reanimation . Daniel visits a nearby sports arena to verify Adams ' story that it was sealed with 2,000 now - zombified civilians inside .
</item 8>

<item 9>
Fear the Walking Dead (season 1)
After Travis and Madison arrive , the zombified Calvin attacks them . Nick runs over Calvin repeatedly with Travis ' truck , and the three watch in disbelief as
the mutilated Calvin is still able to turn his head towards them . `` So Close , Yet So Far '' Adam Davidson Marco Ramirez August 30 , 2015 ( 2015 - 08 - 30 )
8.18 After the run - in with zombified Calvin , Nick , Madison , and Travis choose to flee to the desert , and they want Alicia , Liza , and Chris to follow .
Alicia finds Matt ill in his disheveled house . Travis arrives and sees a bite on Matt , who convinces Alicia to leave . The group returns to Madison 's home to
gather supplies . Nick suffers from heroin withdrawal , so Madison drives to her school to get him Oxycodone . There , she encounters Tobias scavenging food . A
zombified Artie tries to bite Tobias , so Madison kills Artie and drives Tobias home . Chris ' bus is trapped in a traffic jam caused by a zombie shooting by
the police .
</item 9>

<item 10>
Fear the Walking Dead (season 1)
On March 9 , 2015 , AMC announced it had ordered Fear the Walking Dead to series , with a two - season commitment . The second season , comprising 15 episodes ,
premiered on April 10 , 2016 . The season follows a dysfunctional , blended family composed of high school guidance counselor Madison Clark , her English
teacher fiancé Travis Manawa , her daughter Alicia , her drug - addicted son Nick , Travis ' son from a previous marriage , Chris , Chris ' mother Liza Ortiz ,
and others who join their group at the onset of the zombie apocalypse . Contents ( hide ) 1 Cast 1.1 Main cast 1.2 Supporting cast 2 Episodes 3 Production 3.1
Development 3.2 Casting 3.3 Filming 4 Reception 4.1 Critical response 4.2 Ratings 5 Home media releases 6 References 7 External links Cast ( edit ) Main cast (
edit ) Kim Dickens as Madison Clark : A high school guidance counselor , Travis ' girlfriend , Alicia and Nick 's mother , and Chris ' stepmother .
</item 10>
=== END ITEM id=7f39b4fe3eb0f2b8 ===
