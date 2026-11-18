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

=== ITEM id=44f72795f074629d ===
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
Menstruation
Some anthropologists have questioned the energy cost of rebuilding the endometrium every fertility cycle . However , anthropologist Beverly Strassmann has
proposed that the energy savings of not having to continuously maintain the uterine lining more than offsets energy cost of having to rebuild the lining in the
next fertility cycle , even in species such as humans where much of the lining is lost through bleeding ( overt menstruation ) rather than reabsorbed ( covert
menstruation ) . Many have questioned the evolution of overt menstruation in humans and related species , speculating on what advantage there could be to losing
blood associated with dismantling the endometrium , rather than absorbing it , as most mammals do . Humans do , in fact , reabsorb about two - thirds of the
endometrium each cycle . Strassmann asserts that overt menstruation occurs not because it is beneficial in itself . Rather , the fetal development of these
species requires a more developed endometrium , one which is too thick to reabsorb completely . Strassman correlates species that have overt menstruation to
those that have a large uterus relative to the adult female body size .
</item 1>

<item 2>
Menstruation
Scotland is the first country to ban period poverty . In 2015 a petition was put forward in England seeking the elimination of the 5 % `` tampon tax '' . The
government voted in favour but it has yet to be implemented due to Brexit complications . In England , one in ten women between 14 and 21 can not afford
menstrual management products . In the United States , nine states have eliminated the tampon tax , and seven states have introduced legislation . In January
2018 , California rejected a proposal to eliminate tampon tax . Evolution All female placental mammals have a uterine lining that builds up when the animal is
fertile , but it is dismantled when the animal is infertile . Most female mammals have an estrous cycle , yet only primates ( including humans ) , several
species of bats , and elephant shrews have a menstrual cycle . ( Killer whales , Orcinus orca , are one of the few species to experience menopause . ) Some
anthropologists have questioned the energy cost of rebuilding the endometrium every fertility cycle .
</item 2>

<item 3>
Menstruation
Menstrual cups are usually made of silicone and can last 5 years or longer . Reusable cloth pads -- Pads that are made of cotton ( often organic ) , terrycloth
, or flannel , and may be handsewn ( from material or reused old clothes and towels ) or storebought . Padded panties or period - proof underwear -- Reusable
cloth ( usually cotton ) underwear with extra absorbent layers sewn in to absorb flow . Some also use patented technology to be leak resistant , such as the
brand THINX . Sea sponges -- Natural sponges , worn internally like a tampon to absorb menstrual flow . Blanket , towel -- ( also known as a draw sheet ) --
large reusable piece of cloth , most often used at night , placed between legs to absorb menstrual flow . Non-commercial items In developing countries , many
poor women can not afford commercial feminine hygiene products . Instead , they use materials found in the environment or other improvised materials .
</item 3>

<item 4>
Menstruation
This is in part due to the segregation of male and female peers during educational sessions . Failure to teach an accurate understanding of menstruation to
students of all genders has social implications for gendered relationships and the objectification of women 's bodies . Discomfort arises when students do not
have access to the same information , reinforcing the belief `` that menstruation is gross and should be kept hidden '' . Girls are encouraged to conceal the
fact that they may be menstruating in order to be considered desirable . Sexual harassment and teasing about menstruation cause girls anxiety as they must
struggle to ensure that they give no sign of menstruation . Effective educational programs are essential to providing children and adolescents with clear and
accurate information about menstruation . Several education and sexual health experts have studied the key features necessary for such programs . Some experts
maintain that schools are an appropriate place for menstrual education to take place because they are an institution that young people attend consistently .
Schools are intended to expand students ' knowledge and thus serve as an appropriate site for conveying menstrual education . Other experts argue that programs
led by peers or third - party agencies are more effective than those taught in the school classroom .
</item 4>

<item 5>
Menstruation
Menstruation - wikipedia Menstruation Jump to : navigation , search This article is about humans . For other mammals , see Menstruation ( mammal ) . Figure
showing the progression of the menstrual cycle and the different hormones contributing to it . Menstruation , also known as a period or monthly , is the regular
discharge of blood and mucosal tissue ( known as menses ) from the inner lining of the uterus through the vagina . The first period usually begins between
twelve and fifteen years of age , a point in time known as menarche . However , periods may occasionally start as young as eight years old and still be
considered normal . The average age of the first period is generally later in the developing world , and earlier in the developed world . The typical length of
time between the first day of one period and the first day of the next is 21 to 45 days in young women , and 21 to 31 days in adults ( an average of 28 days ) .
Bleeding usually lasts around 2 to 7 days . Menstruation stops occurring after menopause , which usually occurs between 45 and 55 years of age .
</item 5>

<item 6>
Menstruation
Menstrual products ( also called `` feminine hygiene '' products ) are made to absorb or catch menstrual blood . A number of different products are available -
some are disposable , some are reusable . Where women can afford it , items used to absorb or catch menses are usually commercially manufactured products .
Disposable products Disposable sanitary napkin Tampon in plastic applicator Sanitary napkins ( also called sanitary towels or pads ) -- Rectangular pieces of
material worn attached to the underwear to absorb menstrual flow , often with an adhesive backing to hold the pad in place . Disposable pads may contain wood
pulp or gel products , usually with a plastic lining and bleached . Tampons -- Disposable cylinders of treated rayon / cotton blends or all - cotton fleece ,
usually bleached , that are inserted into the vagina to absorb menstrual flow . Disposable menstrual cups made of soft plastic -- A firm , flexible cup - shaped
device worn inside the vagina to collect menstrual flow . Reusable products Menstrual cup Cloth menstrual pad Menstrual cups -- A firm , flexible bell - shaped
device worn inside the vagina to collect menstrual flow . Menstrual cups are usually made of silicone and can last 5 years or longer .
</item 6>

<item 7>
Menstruation
The egg is released from an ovary around day fourteen in the cycle ; the thickened lining of the uterus provides nutrients to an embryo after implantation . If
pregnancy does not occur , the lining is released in what is known as menstruation . Contents ( hide ) 1 Onset and frequency 2 Health effects 2.1 Cramps 2.2
Mood and behavior 2.3 Bleeding 2.4 Menstrual disorders 3 Ovulation suppression 3.1 Birth control 3.2 Breastfeeding 4 Menstrual management 4.1 Disposable
products 4.2 Reusable products 4.3 Non-commercial items 5 Society and culture 5.1 Traditions and taboos 5.1. 1 Islam 5.1. 2 Judaism 5.1. 3 Hinduism 5.1. 4
Christianity 5.1. 5 Secular culture 5.1.
</item 7>

<item 8>
Menstruation
Menstruation : A Cultural History . Palgrave Macmillan . ISBN 1 - 4039 - 3935 - 7 . Retrieved 9 November 2013 . Knight , Chris ( 1995 ) . Blood Relations :
Menstruation and the Origins of Culture . New Haven and London : Yale University . ISBN 0 - 300 - 04911 - 0 . Retrieved 9 November 2013 . External links
Wikiquote has quotations related to : Menstruation Wikimedia Commons has media related to Menstruation . Museum of Menstruation Human physiology of sexual
reproduction Menstrual cycle Menarche Menstruation Follicular phase Ovulation Luteal phase Gametogenesis Spermatogenesis ( spermatogonium spermatocyte spermatid
sperm ) Oogenesis ( oogonium oocyte ootid ovum ) Germ cell ( gonocyte gamete ) Human sexual behavior Sexual arousal Sexual intercourse Masturbation Erection
Orgasm Ejaculation Insemination Fertilisation / Fertility Implantation Pregnancy Postpartum period Mechanics of sex Life span Prenatal development / Sexual
dimorphism / Sexual differentiation ( Feminization Virilization
</item 8>

<item 9>
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
</item 9>

<item 10>
Menstruation
Instead , they use materials found in the environment or other improvised materials . Absorption materials that may be used by women who can not afford anything
else include : sand , ash , small hole in earth , cloth , whole leaf , leaf fibre ( such as water hyacinth , banana , papyrus , cotton fibre ) , paper ( toilet
paper , re-used newspaper , pulped and dried paper ) , animal pelt e.g. goat skin , double layer of underwear , skirt or sari . Society and culture Amra Padatik
India , celebration of Menstrual Hygiene Day in India Traditions and taboos Main article : Culture and menstruation Further information : Menstrual taboo Many
religions have menstruation - related traditions . Islam Islam prohibits sexual contact with women during menstruation in the 2nd chapter of the Quran . Women
are exempted from Salah ( prayer ) . They are also prohibited from fasting and obligatory fasts of Ramadan are to be performed during other days . During menses
pilgrimages are allowed ; however , circumambulation of the Kaaba is prohibited and is to be performed during other times .
</item 10>
=== END ITEM id=44f72795f074629d ===

=== ITEM id=0fb9a56ff2ad3533 ===
Gold sentences and facts:
Sentence 0: All black military pilots who trained in the United States trained at Moton Field , the Tuskegee Army Air Field , and were educated at Tuskegee University , located near Tuskegee , Alabama . The group included five Haitians from the Haitian Air Force , and one pilot from Trinidad. . It also included a Hispanic or Latino airman born in the Dominican Republic .
  Fact 0.0: All black military pilots who trained in the United States trained at Moton Field.
  Fact 0.1: All black military pilots who trained in the United States trained at the Tuskegee Army Air Field.
  Fact 0.2: All black military pilots who trained in the United States were educated at Tuskegee University.
  Fact 0.3: Tuskegee University is located near Tuskegee, Alabama.
  Fact 0.4: The group included five Haitians from the Haitian Air Force.
  Fact 0.5: The group included one pilot from Trinidad.
  Fact 0.6: The group included a Hispanic or Latino airman born in the Dominican Republic.

Retrieved items:
<item 1>
Tuskegee Airmen
Census Bureau reported there were only 124 African - American pilots in the nation . The exclusionary policies failed dramatically when the Air Corps received
an abundance of applications from men who qualified , even under the restrictive requirements . Many of the applicants already had participated in the Civilian
Pilot Training Program , unveiled in late December 1938 ( CPTP ) . Tuskegee University had participated since 1939 . Testing ( edit ) This section does not cite
any sources . Please help improve this section by adding citations to reliable sources . Unsourced material may be challenged and removed . ( January 2017 ) (
Learn how and when to remove this template message ) The U.S. Army Air Corps had established the Psychological Research Unit 1 at Maxwell Army Air Field ,
Montgomery , Alabama , and other units around the country for aviation cadet training , which included the identification , selection , education , and training
of pilots , navigators , and bombardiers .
</item 1>

<item 2>
Tuskegee Airmen
They formed the 332nd Fighter Group and the 477th Bombardment Group of the United States Army Air Forces . The name also applies to the navigators , bombardiers
, mechanics , instructors , crew chiefs , nurses , cooks and other support personnel . All black military pilots who trained in the United States trained at
Moton Field , the Tuskegee Army Air Field , and were educated at Tuskegee University , located near Tuskegee , Alabama . The group included five Haitians from
the Haitian Air Force , and one pilot from Trinidad. . It also included a Hispanic or Latino airman born in the Dominican Republic . Although the 477th
Bombardment Group trained with North American B - 25 Mitchell bombers , they never served in combat . The 99th Pursuit Squadron ( later , 99th Fighter Squadron
) was the first black flying squadron , and the first to deploy overseas ( to North Africa in April 1943 , and later to Sicily and Italy ) . The 332nd Fighter
Group , which originally included the 100th , 301st , and 302nd Fighter Squadrons , was the first black flying group .
</item 2>

<item 3>
Tuskegee Airmen
Coleman Young Incidents Freeman Field Mutiny Aircraft Bell P - 39 Airacobra Curtiss P - 40 Warhawk North American P - 51 Mustang Republic P - 47 Thunderbolt
Media Fictional films The Tuskegee Airmen ( 1995 ) Hart 's War ( 2002 ) Red Tails ( 2012 ) Documentaries Red Tail Reborn ( 2007 ) Flight of the Red Tail ( 2009
) Other Henry Browne , Farmer Wings for This Man `` Wingmen '' Legacy Red Tail Squadron Tuskegee Airmen , Inc Tuskegee University Tuskegee Airmen National
Historic Site Other Maycie Herrington Buford A. Johnson Noel F. Parrish See also African American military history Buffalo Soldier Golden Thirteen 366th
Infantry Regiment U.S. Army Black Panthers United States Colored Troops USAAF Fifteenth Air Force in World War II Stations Tunisia Bizerte Depienne Enfidaville
Massicault Oudna Pont du Fahs Sainte Marie du Zit Soliman Italy Amendola Bari Castelltuccio Cattolica Celone Cerignola Fano Foggia Gioia del Colle Grottaglie
</item 3>

<item 4>
Tuskegee Airmen
The white population of Freeman Field was 250 officers and 600 enlisted men . Superimposed on it were 400 African - American officers and 2,500 enlisted men of
the 477th and its associated units . Freeman Field had a firing range , usable runways , and other amenities useful for training . African - American airmen
would work in proximity with white ones ; both would live in a public housing project adjacent to the base . Colonel Selway turned the non-commissioned officers
out of their club and turned it into a second officers club . He then classified all white personnel as cadre , and all African - Americans as trainees . One
officers club became the cadre 's club . The old Non-Commissioned Officers Club , promptly sarcastically dubbed `` Uncle Tom 's Cabin '' , became the trainee 's
officers club . At least four of the trainees had flown combat in Europe as fighter pilots , and had about four years in service . Four others had completed
training as pilots , bombardiers and navigators , and may have been the only triply qualified officers in the entire Air Corps .
</item 4>

<item 5>
Tuskegee Airmen
This particular P - 51C is part of the Red Tail Project The Stearman Kaydet training aircraft used by the Tuskegee Airmen , bearing the name Spirit of Tuskegee
Portrait of Tuskegee airman Edward M. Thomas by photographer Toni Frissell , March 1945 Before the Tuskegee Airmen , no African - American had been a U.S.
military pilot . In 1917 , African - American men had tried to become aerial observers , but were rejected . African - American Eugene Bullard served in the
French air service during World War I , because he was not allowed to serve in an American unit . Instead , Bullard returned to infantry duty with the French .
The racially motivated rejections of World War I African - American recruits sparked more than two decades of advocacy by African - Americans who wished to
enlist and train as military aviators . The effort was led by such prominent civil rights leaders as Walter White of the National Association for the
Advancement of Colored People , labor union leader A. Philip Randolph , and Judge William H. Hastie .
</item 5>

<item 6>
Tuskegee Airmen
After landing , she cheerfully announced , `` Well , you can fly all right . '' The subsequent brouhaha over the First Lady 's flight had such an impact it is
often mistakenly cited as the start of the CPTP at Tuskegee , even though the program was already five months old . Eleanor Roosevelt used her position as a
trustee of the Julius Rosenwald Fund to arrange a loan of $175,000 to help finance the building of Moton Field . Formation ( edit ) Major James A. Ellison
returns the salute of Mac Ross , as he reviews the first class of Tuskegee cadets ; flight line at U.S. Army Air Corps basic and advanced flying school , with
Vultee BT - 13 trainers in the background , Tuskegee , Alabama , 1941 On 11 September 1941 , the 99th Pursuit Squadron was activated at Chanute Field in Rantoul
, Illinois . A cadre of 271 enlisted men was trained in aircraft ground support trades at Chanute , beginning in July 1941 ; the skills being taught were so
technical that setting up segregated classes was deemed impossible .
</item 6>

<item 7>
Tuskegee Airmen
Finally , on 3 April 1939 , Appropriations Bill Public Law 18 was passed by Congress containing an amendment by Senator Harry H. Schwartz , designating funds
for training African - American pilots . The War Department managed to put the money into funds of civilian flight schools willing to train black Americans .
War Department tradition and policy mandated the segregation of African - Americans into separate military units staffed by white officers , as had been done
previously with the 9th Cavalry , 10th Cavalry , 24th Infantry Regiment and 25th Infantry Regiment . When the appropriation of funds for aviation training
created opportunities for pilot cadets , their numbers diminished the rosters of these older units . In 1941 , the War Department and the Army Air Corps , under
pressure -- three months before its transformation into the USAAF -- constituted the first all - black flying unit , the 99th Pursuit Squadron . Due to the
restrictive nature of selection policies , the situation did not seem promising for African - Americans since , in 1940 , the U.S. Census Bureau reported there
were only 124 African - American pilots in the nation .
</item 7>

<item 8>
Tuskegee Airmen
Some ground crews trained at Mather before rotating to Inglewood . Gunners learned to shoot at Eglin Field , Florida . Bombers - navigators learned their trades
at Hondo Army Air Field and Midland Air Field , Texas or at Roswell , New Mexico . Training of the new African - American crewmen also took place at Sioux Falls
, South Dakota , Lincoln , Nebraska , and Scott Field , Belleville , Illinois . Once trained , the air and ground crews would be spliced into a working unit at
Selfridge . Command difficulties ( edit ) The new group 's first commanding officer was Colonel Robert Selway , who had also commanded the 332nd Fighter Group
before it deployed for combat overseas . Like his ranking officer , Major General Frank O'Driscoll Hunter from Georgia , Selway was a racial segregationist .
Hunter was blunt about it , saying such things as , `` ... racial friction will occur if colored and white pilots are trained together . '' He backed Selway 's
violations of Army Regulation 210 - 10 , which forbade segregation of air base facilities .
</item 8>

<item 9>
Tuskegee Airmen
Later that year , the Air Corps replaced Kimble . His replacement had been the director of instruction at Tuskegee Army Airfield , Major Noel F. Parrish .
Counter to the prevalent racism of the day , Parrish was fair and open - minded and petitioned Washington to allow the Tuskegee Airmen to serve in combat .
Instrument certificate for Tuskegee Airman Robert M. Glass , signed by Parrish The strict racial segregation the U.S. Army required gave way in the face of the
requirements for complex training in technical vocations . Typical of the process was the development of separate African - American flight surgeons to support
the operations and training of the Tuskegee Airmen . Before the development of this unit , no U.S. Army flight surgeons had been black . Training of African -
American men as aviation medical examiners was conducted through correspondence courses until 1943 , when two black physicians were admitted to the U.S. Army
School of Aviation Medicine at Randolph Field , Texas . This was one of the earliest racially integrated courses in the U.S. Army .
</item 9>

<item 10>
Tuskegee Airmen
Psychologists employed in these research studies and training programs used some of the first standardized tests to quantify IQ , dexterity and leadership
qualities to select and train the best - suited personnel for the roles of bombardier , navigator , and pilot . The Air Corps determined that the existing
programs would be used for all units , including all - black units . At Tuskegee , this effort continued with the selection and training of the Tuskegee Airmen
. The War Department set up a system to accept only those with a level of flight experience or higher education which ensured that only the most able and
intelligent African - American applicants were able to join . The first Lady 's flight ( edit ) The budding flight program at Tuskegee received a publicity
boost when First Lady Eleanor Roosevelt inspected it in March 1941 , and flew with African - American chief civilian instructor C. Alfred `` Chief '' Anderson .
Anderson , who had been flying since 1929 , and was responsible for training thousands of rookie pilots , took his prestigious passenger on a half - hour flight
in a Piper J - 3 Cub . After landing , she cheerfully announced , `` Well , you can fly all right . ''
</item 10>
=== END ITEM id=0fb9a56ff2ad3533 ===

=== ITEM id=1f0478b24aa40d7c ===
Gold sentences and facts:
Sentence 0: Philippe Petit ( French pronunciation : ​ ( filip pəti ) ; born 13 August 1949 ) is a French high - wire artist who gained fame for his high - wire walk between the Twin Towers of the World Trade Center in New York City , on the morning of August 7 , 1974 as well as his high wire walk between the Notre Dame cathedral in Paris , 1971 . For his unauthorized feat 400 metres ( 1,000 feet ) above the ground -- which he referred to as `` le coup '' -- he rigged a 200 - kilogram ( 440 - pound ) cable and used a custom - made 8 - metre ( 30 - foot ) long , 25 - kilogram ( 55 - pound ) balancing pole . He performed for 45 minutes , making eight passes along the wire . The following week , he celebrated his 25th birthday . All charges were dismissed in exchange for him doing a performance in Central Park for children .
  Fact 0.0: Philippe Petit is French.
  Fact 0.1: Philippe Petit was born on 13 August 1949.
  Fact 0.2: Philippe Petit is a high-wire artist.
  Fact 0.3: Philippe Petit gained fame for his high-wire walk between the Twin Towers of the World Trade Center in New York City.
  Fact 0.4: Philippe Petit's Twin Towers walk took place on the morning of 7 August 1974.
  Fact 0.5: Philippe Petit made a high-wire walk between the Notre Dame cathedral in Paris in 1971.
  Fact 0.6: Philippe Petit's feat was unauthorized.
  Fact 0.7: Philippe Petit's feat took place 400 metres (1,000 feet) above the ground.
  Fact 0.8: Philippe Petit referred to the feat as 'le coup'.
  Fact 0.9: Philippe Petit rigged a 200-kilogram (440-pound) cable.
  Fact 0.10: Philippe Petit used a custom-made balancing pole.
  Fact 0.11: The balancing pole was 8 metres (30 feet) long.
  Fact 0.12: The balancing pole weighed 25 kilograms (55 pounds).
  Fact 0.13: Philippe Petit performed for 45 minutes.
  Fact 0.14: Philippe Petit made eight passes along the wire.
  Fact 0.15: Philippe Petit celebrated his 25th birthday the following week.
  Fact 0.16: All charges against Philippe Petit were dismissed.
  Fact 0.17: The charges were dismissed in exchange for Philippe Petit doing a performance in Central Park for children.

Retrieved items:
<item 1>
At 18, Philippe Petit became fascinated by the proposed Twin Towers after seeing their plans in a magazine at a dentist’s office in 1968. He spent six years
studying the buildings and planning his secret high-wire walk, while honing his skills through public performances. In 1971, he walked between Notre-Dame’s
towers during priestly ordinations; in 1973, he crossed a wire between the north pylons of Sydney Harbour Bridge. Office workers, construction crews, and
</item 1>

<item 2>
Philippe Petit honored those killed and urged rebuilding the Twin Towers, promising to walk between them again in celebration. They were replaced by a different
complex, making such a return impossible. His honors include the James Park Morton Interfaith Award, Streb Action Maverick Award, and Byrdcliffe Award. A
performance list mentions his 1971 Vallauris walk for Pablo Picasso’s 90th birthday and a Notre Dame Cathedral entry; the section warns that citations are
needed.
</item 2>

<item 3>
Philippe Petit
In planning for the Twin Towers walk , Petit had to learn how to accommodate such issues as the swaying of the high towers due to wind , which was part of their
design ; effects of wind and weather on the wire at that height , how to rig a 200 ft ( 61 m ) steel cable across the 138 ft ( 42 m ) gap between the towers (
at a height of 1,368 ft ( 417 m ) ) , and how to gain entry with his collaborators , first to scope out the conditions and lastly , to stage the project . They
had to get heavy equipment to the rooftops . He traveled to New York on numerous occasions to make first - hand observations . Since the towers were still under
construction , Petit and one of his collaborators , New York - based photographer Jim Moore , rented a helicopter to take aerial photographs of the buildings .
Jean - François and Jean - Louis helped him practice in a field in France , and accompanied him to take part in the final rigging of the project , as well as to
photograph it . Francis Brunn , a German juggler , provided financial support for the proposed project and its planning .
</item 3>

<item 4>
“Philippe Petit MSA – The Man Who Walked Between the Towers” was co-produced by Michael Sporn Animation and Weston Woods Studios. Philippe Petit’s signature
appears in 1980s photographs of his high-wire crossing between the Twin Towers. The material also references his TED2012 talk, “The Journey Across the High
Wire,” filmed in March and posted in May 2012; the cited Wikipedia link is incomplete.
</item 4>

<item 5>
After Philippe Petit’s 1974 high-wire walk between the Twin Towers, the Port Authority gave him a lifetime observation-deck pass, and he signed a nearby steel
beam. His performance helped popularize the initially disliked towers, criticized as ugly, oversized, and difficult to rent. It inspired Mordicai Gerstein’s
2003 Caldecott-winning picture book and a 2005 award-winning animated film. James Marsh’s 2008 documentary *Man on Wire* won both the
</item 5>

<item 6>
Philippe Petit’s 1974 walk between the Twin Towers inspired numerous works: songs by Ste van Holm, The Low Anthem, Cormorant, Howard Moss, and The Script;
Incubus used his image for its 2011 album *If Not Now, When?*; and Colum McCann’s novel *Let the Great World Spin* opens with the walk. Further reading includes
Mordicai Gerstein’s *The Man Who Walked Between the Towers*.
</item 6>

<item 7>
French high-wire artist Philippe Petit spent six years planning a secret 1974 walk between New York’s Twin Towers. He surveyed the buildings, studied wind and
sway, practiced earlier walks at Notre-Dame and Sydney Harbour Bridge, and used disguises and forged IDs to smuggle in equipment and rig a 200-foot cable across
the 138-foot gap, 1,368 feet above ground. On August 7, he walked, danced, and knelt across it eight times over 45
</item 7>

<item 8>
Philippe Petit
The Port Authority of New York and New Jersey gave Petit a lifetime pass to the Twin Towers ' Observation Deck . He autographed a steel beam close to the point
where he began his walk . Petit 's high - wire walk is credited with bringing the Twin Towers much needed attention and even affection , as they initially had
been unpopular . Critics such as historian Lewis Mumford had regarded them as ugly and utilitarian in design , and too large a development for the area . The
Port Authority was having trouble renting out all of the office space . Mordicai Gerstein wrote and illustrated a children 's book , The Man Who Walked Between
The Towers ( 2003 ) , which won a Caldecott Medal for his art . It was adapted and produced as an animated short film by the same title , directed by Michael
Sporn and released in 2005 , which won several awards . The documentary film Man on Wire ( 2008 ) by UK director James Marsh is about Petit and his 1974 WTC
performance ; it won both the World Cinema Jury and Audience awards at the Sundance Film Festival 2008 .
</item 8>

<item 9>
Philippe Petit
MSA - The Man Who Walked Between The Towers . Co-produced by Michael Sporn Animation and Weston Woods Studios Philippe Petit Signature visible in the 1980s
photos of Philippe Petit crossing the Twin Towers along with others Philippe Petit : The journey across the high wire , TED2012 , Filmed Mar 2012 , Posted May
2012 . Retrieved from `` https://en.wikipedia.org/w/index.
</item 9>

<item 10>
Philippe Petit, a French high-wire artist, secretly rigged a cable between the World Trade Center’s Twin Towers and walked across it in 1974. The references
document the feat and his career through news and magazine articles, the documentary *Man on Wire*, his memoir *To Reach the Clouds*, and material on other
performances, including at Notre Dame and Sydney Harbour Bridge.
</item 10>
=== END ITEM id=1f0478b24aa40d7c ===

=== ITEM id=b2d661f30ef7793a ===
Gold sentences and facts:
Sentence 0: The American entry into World War I came in April 1917 , after more than two and a half years of efforts by President Woodrow Wilson to keep the United States out of the war . Apart from an Anglophile element urging early support for the British , American public opinion reflected that of the president : the sentiment for neutrality was particularly strong among Irish Americans , German Americans and Scandinavian Americans , as well as among church leaders and among women in general . On the other hand , even before World War I had broken out , American opinion had been more negative toward Germany than towards any other country in Europe . Over time , especially after reports of atrocities in Belgium in 1914 and following the sinking of the passenger liner RMS Lusitania in 1915 , the American people increasingly came to see Germany as the aggressor in Europe .
  Fact 0.0: The American entry into World War I came in April 1917.
  Fact 0.1: The American entry into World War I came after more than two and a half years of efforts by President Woodrow Wilson to keep the United States out of the war.
  Fact 0.2: Apart from an Anglophile element urging early support for the British, American public opinion reflected that of the president.
  Fact 0.3: The sentiment for neutrality was particularly strong among Irish Americans.
  Fact 0.4: The sentiment for neutrality was particularly strong among German Americans.
  Fact 0.5: The sentiment for neutrality was particularly strong among Scandinavian Americans.
  Fact 0.6: The sentiment for neutrality was particularly strong among church leaders.
  Fact 0.7: The sentiment for neutrality was particularly strong among women in general.
  Fact 0.8: Before World War I had broken out, American opinion had been more negative toward Germany than toward any other country in Europe.
  Fact 0.9: Over time, the American people increasingly came to see Germany as the aggressor in Europe.
  Fact 0.10: The shift toward seeing Germany as the aggressor was especially driven by reports of atrocities in Belgium in 1914.
  Fact 0.11: The shift toward seeing Germany as the aggressor was especially driven by the sinking of the passenger liner RMS Lusitania in 1915.

Retrieved items:
<item 1>
American entry into World War I
252 -- 282 . ^ Jump up to : H.J. Res. 169 : Declaration of War with Austria - Hungary , WWI , United States Senate ^ Jump up to : Jennifer K. Elsea ; Matthew C.
Weed ( April 18 , 2014 ) . `` Declarations of War and Authorizations for the Use of Military Force : Historical Background and Legal Implications '' ( PDF ) .
p. 9 . Archived from the original ( PDF ) on August 10 , 2006 . Retrieved July 21 , 2014 . Jump up ^ Eric W. Osborne , Britain 's economic blockade of Germany ,
1914 -- 1919 ( 2004 ) Jump up ^ Ernest May , The World War and American Isolation , 1914 -- 1917 ( 1959 ) p. 115 quote from Dec 1914 . Jump up ^ C.R.M.F.
Cruttwell , A history of the Great War , 1914 -- 1918 ( 2007 ) p. 191 Jump up ^ Edward House , The intimate papers of Colonel House : Vol 2 ( 1928 ) p.
</item 1>

<item 2>
American entry into World War I
Germany also made a secret offer to help Mexico regain territories lost in the Mexican -- American War in an encoded telegram known as the Zimmermann Telegram ,
which was intercepted by British Intelligence . Publication of that communique outraged Americans just as German U-boats started sinking American merchant ships
in the North Atlantic . Wilson then asked Congress for `` a war to end all wars '' that would `` make the world safe for democracy '' , and Congress voted to
declare war on Germany on April 6 , 1917 . On December 7 , 1917 , the U.S. declared war on Austria - Hungary . U.S. troops began arriving on the Western Front
in large numbers in 1918 . Contents ( hide ) 1 Main issues 1.1 Naval blockade 1.1. 1 The strategy behind the blockade 1.2 Business considerations 1.3 Views of
the elites 2 Public opinion 2.1 Parties 2.2 Workers , farmers , and African Americans 2.3 South 2.4 German Americans 2.5 Christian churches and pacifists 2.6
Jewish Americans 2.7 Irish - Americans 2.
</item 2>

<item 3>
American entry into World War I
On April 22 , the German Imperial Embassy warned U.S. citizens against boarding vessels to Britain , which would have to face German attack . On May 7 , Germany
torpedoed the British passenger liner RMS Lusitania , sinking her . This act of aggression caused the loss of 1,198 civilian lives , including 128 Americans .
The sinking of a large , unarmed passenger ship , combined with the previous stories of atrocities in Belgium , shocked Americans and turned public opinion
hostile to Germany , although not yet to the point of war . Wilson issued a warning to Germany that it would face `` strict accountability '' if it sank more
neutral U.S. passenger ships . Berlin acquiesced , ordering its submarines to avoid passenger ships . By January 1917 , however , Field Marshal Paul von
Hindenburg and General Erich Ludendorff decided that an unrestricted submarine blockade was the only way to achieve a decisive victory . They demanded that
Kaiser Wilhelm order unrestricted submarine warfare be resumed . Germany knew this decision meant war with the United States , but they gambled that they could
win before America 's potential strength could be mobilized .
</item 3>

<item 4>
American entry into World War I
Germany 's U-boat warfare was paramount . Had it not been for Berlin 's announcement of January 31 , the president probably would not have issued a call to arms
. Jump up ^ Justus D. Doenecke , Nothing Less Than War : A New History of America 's Entry into World War I ( 2011 ) ch 10 ^ Jump up to : May , Ernest R. ( 1966
) . The World War and American isolation , 1914 -- 1917 . Jump up ^ Knock , Thomas J. ( 1995 ) . To End All Wars : Woodrow Wilson and the Quest for a New World
Order . Jump up ^ Syrett , Harold C. ( 1945 ) . `` The Business Press and American Neutrality , 1914 -- 1917 '' . Mississippi Valley Historical Review . 32 ( 2
) : 215 -- 230 . doi : 10.2307 / 1898209 . JSTOR 1898209 .
</item 4>

<item 5>
American entry into World War I
Sinking of American merchant ships ( edit ) In early 1917 Berlin forced the issue . Its declared decision on 31 January 1917 to target neutral shipping in a
designated war - zone became the immediate cause of the entry of the United States into the war . Five American merchant ships went down in March . Outraged
public opinion now overwhelmingly supported Wilson when he asked Congress for a declaration of war on April 2 , 1917 . Public opinion , moralism , and national
interest ( edit ) Historians such as Ernest R. May have approached the process of American entry into the war as a study in how public opinion changed radically
in three years ' time . In 1914 most Americans called for neutrality , seeing the war a dreadful mistake and were determined to stay out . By 1917 the same
public felt just as strongly that going to war was both necessary and wise . Military leaders had little to say during this debate , and military considerations
were seldom raised . The decisive questions dealt with morality and visions of the future . The prevailing attitude was that America possessed a superior moral
position as the only great nation devoted to the principles of freedom and democracy .
</item 5>

<item 6>
American entry into World War I
The solution was a third route , a `` peace without victory '' , according to Wilson . On April 6 , 1917 , Congress declared war . In the Senate , the
resolution passed 82 to 6 , with Senators Harry Lane , William J. Stone , James Vardaman , Asle Gronna , Robert M. La Follette , Sr. , and George W. Norris
voting against it . In the House , the declaration passed 373 to 50 , with Claude Kitchin , a senior Democrat , notably opposing it . Another opponent was
Jeannette Rankin , who alone voted against entry into both World War I and World War II . Nearly all of the opposition came from the West and the Midwest .
Austria - Hungary ( edit ) The United States Senate , in a 74 to 0 vote , declared war on Austria - Hungary on December 7 , 1917 , citing Austria - Hungary 's
severing of diplomatic relations with the United States , its use of unrestricted submarine warfare and its alliance with Germany . The declaration passed in
the United States House of Representatives by a vote of 365 to 1 .
</item 6>

<item 7>
American entry into World War I
International Encyclopedia of the First World War . NY Times main headline , April 2 , 1917 , President Calls for War Declaration , Stronger Navy , New Army of
500,000 Men , Full Cooperation With Germany 's Foes President Wilson 's War Address Map of Europe at the time of the US declaration of war on Germany at
omniatlas.com World War I : Declarations of War from Around the Globe - How America Entered the Great War Today in History : U.S.
</item 7>

<item 8>
American entry into World War I
After February 1 , the communique noted , German U-boats would sink without warning belligerent and neutral ships found in a designated zone comprising waters
around Great Britain , France , and Italy , and in the eastern Mediterranean . The Admiralty made one minor exception : it would permit one American steamer a
week to sail between New York and Falmouth ( ... ) . Initially Germany would grant a period of grace , during which its submarines would not harm neutral ships
that either were en route to the war zone or had already arrived . Jump up ^ Compare : Doenecke , Justus D. ( 2011 ) . Nothing Less Than War : A New History of
America 's Entry Into World War I. Studies in Conflict , Diplomacy and Peace Series . University Press of Kentucky . p. 286 . ISBN 978 - 0 - 8131 - 3002 - 6 .
Retrieved 2015 - 12 - 07 . Several factors led to Wilson 's choice . Germany 's U-boat warfare was paramount .
</item 8>

<item 9>
American entry into World War I
Bourne believes elites knew full well what going to war would entail and the price in American lives it would cost . If American elites could portray the United
States ' role in the war as noble , they could convince the generally isolationist American public war would be acceptable . Above all , American attitudes
towards Germany focused on the U-boats ( submarines ) , which sank the Lusitania in 1915 and other passenger ships `` without warning '' . That appeared to
Americans as an unacceptable challenge to America 's rights as a neutral country , and as an unforgivable affront to humanity . After repeated diplomatic
protests , Germany agreed to stop . But in 1917 the Germany military leadership decided that `` military necessity '' dictated the unrestricted use of their
submarines . The Kaiser 's advisors felt America was enormously powerful economically but too weak militarily to make a difference . Declaration of war ( edit )
Germany ( edit ) On April 2 , 1917 , Wilson asked a special joint session of Congress to declare war on the German Empire , stating , `` We have no selfish ends
to serve '' .
</item 9>

<item 10>
American entry into World War I
However , they overestimated how many ships they could sink and thus the extent Britain would be weakened . Finally , they did not foresee that convoys could
and would be used to defeat their efforts . They believed that the United States was so weak militarily that it could not be a factor on the Western Front for
more than a year . The civilian government in Berlin objected , but the Kaiser sided with his military . Business considerations ( edit ) The beginning of war
in Europe coincided with the end of the Recession of 1913 -- 1914 in America . Exports to belligerent nations rose rapidly over the first four years of the War
from $824.8 million in 1913 to $2.25 billion in 1917 . Loans from American financial institutions to the Allied nations in Europe also increased dramatically
over the same period . Economic activity towards the end of this period boomed as government resources aided the production of the private sector . Between 1914
and 1917 , industrial production increased 32 % and GNP increased by almost 20 % . The improvements to industrial production in the United States outlasted the
war .
</item 10>
=== END ITEM id=b2d661f30ef7793a ===

=== ITEM id=5427ed79c07572f3 ===
Gold sentences and facts:
Sentence 0: John Ernest Crawford ( born March 26 , 1946 ) is an American character actor , singer , and musician . At age 12 , Crawford rose to fame for playing Mark McCain , the son of Lucas McCain ( played by Chuck Connors ) , in the popular ABC Western series , The Rifleman , which originally aired from 1958 to 1963 . Crawford first performed before a national audience as a Mouseketeer .
  Fact 0.0: John Ernest Crawford was born March 26, 1946.
  Fact 0.1: John Ernest Crawford is an American character actor.
  Fact 0.2: John Ernest Crawford is an American singer.
  Fact 0.3: John Ernest Crawford is an American musician.
  Fact 0.4: At age 12, John Ernest Crawford rose to fame for playing Mark McCain.
  Fact 0.5: Mark McCain is the son of Lucas McCain.
  Fact 0.6: Lucas McCain was played by Chuck Connors.
  Fact 0.7: Crawford played Mark McCain in the popular ABC Western series The Rifleman.
  Fact 0.8: The Rifleman originally aired from 1958 to 1963.
  Fact 0.9: John Ernest Crawford first performed before a national audience as a Mouseketeer.

Retrieved items:
<item 1>
Johnny Crawford
Johnny Crawford - wikipedia Johnny Crawford Jump to : navigation , search For the Canadian ice hockey player and coach , see Johnny Crawford ( ice hockey ) .
Johnny Crawford Johnny Crawford in The Rifleman ( 1961 ) John Ernest Crawford ( 1946 - 03 - 26 ) March 26 , 1946 ( age 72 ) Los Angeles , California , U.S.
Occupation Actor , singer , musician , band leader Years active 1954 -- 96 Spouse ( s ) Charlotte Crawford ( 1995 -- present ) ( separated ) Relatives Robert L.
Crawford , Jr. ( brother ) Website Johnny Crawford fansite Johnny Crawford and Chuck Connors in The Rifleman ( 1960 ) John Ernest Crawford ( born March 26 ,
1946 ) is an American character actor , singer , and musician . At age 12 , Crawford rose to fame for playing Mark McCain , the son of Lucas McCain ( played by
Chuck Connors ) , in the popular ABC Western series , The Rifleman , which originally aired from 1958 to 1963 . Crawford first performed before a national
audience as a Mouseketeer .
</item 1>

<item 2>
Johnny Crawford
The third pilot , which was made as an episode of Dick Powell 's Zane Grey Theater , was picked up by ABC and the first season of The Rifleman began filming in
July 1958 . Crawford was nominated for an Emmy Award , at age 13 , for his role as Mark McCain , the son of Lucas McCain , played by Chuck Connors , in the Four
Star Television series The Rifleman , which originally aired from 1958 to 1963 . Throughout The Rifleman 's five seasons , a remarkable on - screen chemistry
existed between Connors and Crawford in the depiction of their father - son relationship . They were still close friends when Connors died on November 10 , 1992
, and Crawford gave a eulogy at Connors ' memorial service . During the late 1950s and early 1960s , Crawford had wide popularity with American teenagers and a
recording career that generated four Billboard Top 40 hits , including the single , `` Cindy 's Birthday '' , which peaked at number 8 , in 1962 .
</item 2>

<item 3>
Graph query row:
entity: Johnny Crawford
relationship: RELATED_TO
relationship_detail: None
role: None
context: None
fact: None
connected_entity: Mark McCain
</item 3>

<item 4>
Johnny Crawford
His first important break as an actor followed with the title role in a Lux Video Theatre production of `` Little Boy Lost '' , a live NBC broadcast on March 15
, 1956 . He also appeared in the popular Western series The Lone Ranger , in 1956 , in one of the few color episodes of that series . Following that performance
, the young actor worked steadily with many seasoned actors and directors . Freelancing for 2 1 / 2 years , he accumulated almost 60 television credits ,
including featured roles in three episodes of NBC 's The Loretta Young Show and an appearance as Manuel in , `` I Am an American '' , an episode of the
syndicated crime drama Sheriff of Cochise . By the spring of 1958 , he had also performed 14 demanding roles in live teleplays for NBC 's Matinee Theatre ,
appeared on CBS 's sitcom , Mr. Adams and Eve , in the Wagon Train episode `` The Sally Potter Story '' ( in which Martin Milner also appeared ) and on the
syndicated series , Crossroads , Sheriff of Cochise , and Whirlybirds , and made three pilots of TV series .
</item 4>

<item 5>
Johnny Crawford
He played a young deputy Billy Norris , in The Big Valley episode `` The Other Face Of Justice '' in 1969 . While enlisted in the United States Army for two
years , Crawford worked on training films as a production coordinator , assistant director , script supervisor , and occasional actor . His rank was sergeant at
the time of his honorable discharge in December 1967 . In 1968 , Crawford played a soldier wanted for murder in `` By the Numbers '' , an episode of the popular
TV series Hawaii Five - O . The Resurrection of Broncho Billy was a USC student film Crawford agreed to do as a favor to his close friend , producer John
Longenecker . It won the 1970 Academy Award for Best Live Action Short Subject . The Naked Ape was a partially animated 1973 feature film starring Crawford and
Victoria Principal , produced by Hugh Hefner . In 1976 , Johnny Crawford co-starred as Ben Shelby in the tenth episode of season three of Little House on the
Prairie , `` The Hunters '' . In 1985 , Crawford appeared as Noah Paisley , sheriff 's deputy , in an episode of Murder , She Wrote .
</item 5>

<item 6>
Jack McCoy
He also revealed that his father was a racist who once hit him for dating a Polish girl . McCoy disliked his father , calling him a `` son - of - a-bitch '' ;
however , he admits he could have easily become like him . While not a nationalist , he cares enough about his heritage to be offended by a lawyer 's
insinuation that two murder suspects committed the crime because of their `` Irish temper '' . McCoy has been divorced twice ( one ex-wife having been a former
assistant ) and has an adult daughter , Rebecca , with Ellen . One of his ex-wives left him because he worked too many late nights . A gossip columnist writes
that McCoy has not seen or spoken to his daughter since 1997 , and McCoy receives an envelope containing pictures of his daughter . He does not open the
envelope ; rather , he places it in his bottom left desk drawer , next to a bottle of Jim Beam . In `` Fallout '' , the last scene shows McCoy meeting his
daughter at a restaurant .
</item 6>

<item 7>
Johnny Crawford
Jump up ^ `` Johnny Crawford '' . Scott Stander . Retrieved January 25 , 2014 . Jump up ^ `` Angry Woman -- Victoria Jackson on `` Tonight Show '' b / w `` .
Youtube . Retrieved January 25 , 2014 . Biography portal California portal Los Angeles portal Film portal Television portal Music portal United States Army
portal External links ( edit ) Johnny Crawford on The Rifleman Broncho Billy short film Johnny Crawford 's MySpace site Johnny Crawford on IMDb VIAF : 63364407
LCCN : no91003539 ISNI : 0000 0001 1526 4402 GND : 142142425 SUDOC : 079205798 BNF : cb11219552n ( data ) MusicBrainz : bf40b9eb - 068a - 4bdd - be46 -
ec0b5bd7a177 BNE : XX4832002 SNAC : w6sk5xbc Retrieved from `` https://en.wikipedia.org/w/index.
</item 7>

<item 8>
The Dukes of Hazzard (film)
Gainey as Sheriff Rosco P. Coltrane Michael Weston as Deputy Enos Strate Lynda Carter as Pauline James Roday as Billy Prickett Kevin Heffernan as Derek `` Sheev
'' Sheevington Nikki Griffin as Katie - Lynn Johnson Jacqui Maxwell as Annette Alice Greczyn as Laurie Pullman Junior Brown as The Balladeer ( narrator ) Joe
Don Baker as Governor Jim Applewhite Barry Corbin as Bill Pullman Andrew Prine as Angry Man Randy Orton as Bo Duke ( Stunt double ) Cameos ( edit ) All five
members of the comedy film troupe Broken Lizard make appearances in the film , classified as cameos , except for Kevin Heffernan , who had a larger speaking
role ( Sheev ) . Broken Lizard cameos Steve Lemme appears as Jimmy Pullman , the son of Bill Pullman , in an early car chase scene in which he accidentally
shoots the inside of his father 's truck . Jay Chandrasekhar and Erik Stolhanske reprise their roles as Ramathorn and Rabbit from the Broken Lizard comedy ,
Super Troopers .
</item 8>

<item 9>
Rio Bravo (film)
In this film , Robert Mitchum played the Dean Martin role , Arthur Hunnicutt the Walter Brennan character , and James Caan the Ricky Nelson role . Hawks again
named the Nelson / Caan character after a state ( in this case , Mississippi ) and in a wry , humorous twist on the original film , Hawks made him inept with
firearms , but skilled with a knife . The second remake , Rio Lobo , was made in 1970 with a plot much further off the original mold , starting with the absence
of a lawman - turned - drunkard character . This began with a Confederate train robbery of a Union gold shipment during the American Civil War , then moved to a
postwar Texas county thoroughly controlled by a rich , arrogant rancher . The heroes , with the exception of an old man similar to Brennan and Hunnicutt 's
characters in the previous pictures ( Jack Elam here ) , were complete outsiders . Along with Wayne and Elam , this movie starred Mexican film star Jorge Rivero
( as Frenchie ) , Christopher Mitchum ( Robert Mitchum 's son ) , and Jennifer O'Neill .
</item 9>

<item 10>
Johnny Crawford
p . A4 . Retrieved January 25 , 2014 . Jump up ^ `` Johnny Crawford ... ' the Son of Rifleman ' '' . TV Week Magazine . The Evening Independent . St. Petersburg
, Florida . May 24 , 1959 . p. 7 . Retrieved January 25 , 2014 . Jump up ^ `` Answers : Johnny Crawford A Mixture '' . The Evening Independent . St. Petersburg
, Florida . November 20 , 1963 . p . A9 . Retrieved January 25 , 2014 . His mother is from Belgium . Jump up ^ 1959 Emmys Jump up ^ `` '' A Very Bright Boy ''
on The Donna Reed Show , December 21 , 1961 `` . Internet Movie Data Base . Retrieved January 14 , 2013 . Jump up ^ `` Johnny Crawford : The Naked Ape '' . Mary
Anderson . Retrieved January 25 , 2014 . Jump up ^ `` Sweepin ' the Clouds Away '' , cduniverse.com ; accessed April 11 , 2015 .
</item 10>
=== END ITEM id=5427ed79c07572f3 ===

=== ITEM id=18f2599731c9ee4a ===
Gold sentences and facts:
Sentence 0: In 1910 , in conjunction with advocacy work led by Florence Kelley of the National Consumer League , the Women 's Trade Union League ( WTLU ) of Massachusetts under the leadership of Elizabeth Evans took up the cause of minimum wage legislation in Massachusetts . Over the next two years , a coalition of social reform groups and labor advocates in Boston pushed for minimum wage legislation in the state . In 1912 , Massachusetts passed the first American minimum wage legislation , which established a state commission for recommending non-compulsory minimum wages for women and children . Within eight years , at least thirteen U.S. states and the District of Columbia would pass minimum wage laws , with pressure being placed on state legislatures by the National Consumers League in a coalition with other women 's voluntary associations and organized labor . The United States Supreme Court of the Lochner era consistently invalidated compulsory minimum wage laws . Advocates for these minimum wage laws hoped that they would be upheld under the precedent of Muller v. Oregon , which upheld maximum working hours laws for women on the grounds that women required special protection that men did not . However , the Court did not extend this principle to minimum wage laws , considering the latter as interfering with the ability of employers to freely negotiate wage contracts with employees .
  Fact 0.0: In 1910, the Women's Trade Union League (WTLU) of Massachusetts, under the leadership of Elizabeth Evans, took up the cause of minimum wage legislation in Massachusetts.
  Fact 0.1: The 1910 effort was in conjunction with advocacy work led by Florence Kelley of the National Consumer League.
  Fact 0.2: Over the next two years, a coalition of social reform groups and labor advocates in Boston pushed for minimum wage legislation in the state.
  Fact 0.3: In 1912, Massachusetts passed the first American minimum wage legislation.
  Fact 0.4: The 1912 Massachusetts law established a state commission for recommending non-compulsory minimum wages for women and children.
  Fact 0.5: Within eight years, at least thirteen U.S. states and the District of Columbia would pass minimum wage laws.
  Fact 0.6: Pressure on state legislatures came from the National Consumers League, in a coalition with other women's voluntary associations and organized labor.
  Fact 0.7: The United States Supreme Court of the Lochner era consistently invalidated compulsory minimum wage laws.
  Fact 0.8: Advocates for minimum wage laws hoped they would be upheld under the precedent of Muller v. Oregon.
  Fact 0.9: Muller v. Oregon upheld maximum working hours laws for women on the grounds that women required special protection that men did not.
  Fact 0.10: The Court did not extend the Muller v. Oregon principle to minimum wage laws.
  Fact 0.11: The Court considered minimum wage laws as interfering with the ability of employers to freely negotiate wage contracts with employees.
Sentence 1: In 1933 , the Roosevelt administration made the first attempt at establishing a national minimum wage , when a $0.25 per hour standard was set as part of the National Industrial Recovery Act . However , in the 1935 court case Schechter Poultry Corp. v. United States ( 295 U.S. 495 ) , the US Supreme Court declared the act unconstitutional , and the minimum wage was abolished . In 1938 , the minimum wage was re-established pursuant to the Fair Labor Standards Act , once again at $0.25 per hour ( $4.78 in 2017 dollars ) . In 1941 , the Supreme Court upheld the Fair Labor Standards Act in United States v. Darby Lumber Co. , holding that Congress had the power under the Commerce Clause to regulate employment conditions .
  Fact 1.0: In 1933, the Roosevelt administration made the first attempt at establishing a national minimum wage.
  Fact 1.1: The 1933 minimum wage attempt set a $0.25 per hour standard as part of the National Industrial Recovery Act.
  Fact 1.2: In the 1935 court case Schechter Poultry Corp. v. United States (295 U.S. 495), the US Supreme Court declared the National Industrial Recovery Act unconstitutional.
  Fact 1.3: The national minimum wage was abolished after the Schechter Poultry ruling.
  Fact 1.4: In 1938, the minimum wage was re-established under the Fair Labor Standards Act.
  Fact 1.5: The 1938 minimum wage was $0.25 per hour.
  Fact 1.6: The 1938 rate of $0.25 per hour equals $4.78 in 2017 dollars.
  Fact 1.7: In 1941, the Supreme Court upheld the Fair Labor Standards Act in United States v. Darby Lumber Co.
  Fact 1.8: In United States v. Darby Lumber Co., the Court held that Congress had the power under the Commerce Clause to regulate employment conditions.

Retrieved items:
<item 1>
Minimum wage in the United States
( 1913 ) . `` The Minimum Wage in Great Britain and Australia '' . The Annals of the American Academy of Political and Social Science . 48 : 26 . Jump up ^
Vivien. , Hart , ( 2001 ) . Bound by Our Constitution : Women , Workers , and the Minimum Wage . Princeton : Princeton University Press . p. 67 . ISBN
9781400821563 . OCLC 700688619 . Jump up ^ Webb , Sidney ( 1912 ) . `` The Economic Theory of a Legal Minimum Wage '' . Journal of Political Economy. 20 ( 10 )
: 973 . Jump up ^ Starr , Gerald ( 1993 ) . Minimum wage fixing : an international review of practices and problems ( 2nd impression ( with corrections ) ed . )
. Geneva : International Labour Office . p. 1 . ISBN 9789221025115 . Jump up ^ Hammond , Matthew B. ( 1913 ) . `` The Minimum Wage in Great Britain and
Australia '' . The Annals of the American Academy of Political and Social Science .
</item 1>

<item 2>
Minimum wage in the United States
May 2006 . Retrieved 2013 - 10 - 06 . ^ Jump up to : `` National Occupational Employment and Wage Estimates United States '' . US Bureau of Labor Statistics .
May 2008 . Retrieved 2013 - 10 - 06 . Jump up ^ `` National Occupational Employment and Wage Estimates United States '' . US Bureau of Labor Statistics . May
2009 . Retrieved 2013 - 10 - 06 . External links ( edit ) Minimum wage in the United States at Curlie ( based on DMOZ ) Federal Minimum Wage . United States
Department of Labor Wage and Hour Division . Minimum Wages for Tipped Employees . United States Department of Labor Wage and Hour Division . History of Federal
Minimum Wage United States Department of Labor Wage and Hour Division . U.S. Minimum Wage History . Oregon State University -- Wealth and Poverty ( Anth 484 ) .
Last updated December 26 , 2012 . Maps and charts of minimum wage vs. housing costs - National Low Income Housing Coalition ( advocacy group ) United States
state - related lists List of states and territories of the United States Demographics
</item 2>

<item 3>
Minimum wage in the United States
Minimum wage in the United States - wikipedia Minimum wage in the United States Jump to : navigation , search Minimum wage by U.S. state and territory , as of
January 1 , 2018 . Versus the federal rate : Higher Same Lower No state minimum wage law The minimum wage in the United States is set by US labor law and a
range of state and local laws . Employers generally have to pay workers the highest minimum wage prescribed by federal , state , and local law . Since July 24 ,
2009 , the federal government has mandated a nationwide minimum wage of $7.25 per hour . As of January 2018 , there were 29 states with a minimum wage higher
than the federal minimum . From 2017 to 2018 , eight states increased their minimum wage levels through automatic adjustments , while increases in eleven other
states occurred through referendum or legislative action . Washington has the highest state minimum wage at $11.50 per hour . Using 2017 inflation - adjusted
dollars , the federal minimum wage peaked at $11.53 per hour in 1968 .
</item 3>

<item 4>
Minimum wage in the United States
However , the Court did not extend this principle to minimum wage laws , considering the latter as interfering with the ability of employers to freely negotiate
wage contracts with employees . In 1933 , the Roosevelt administration made the first attempt at establishing a national minimum wage , when a $0.25 per hour
standard was set as part of the National Industrial Recovery Act . However , in the 1935 court case Schechter Poultry Corp. v. United States ( 295 U.S. 495 ) ,
the US Supreme Court declared the act unconstitutional , and the minimum wage was abolished . In 1938 , the minimum wage was re-established pursuant to the Fair
Labor Standards Act , once again at $0.25 per hour ( $4.78 in 2017 dollars ) . In 1941 , the Supreme Court upheld the Fair Labor Standards Act in United States
v. Darby Lumber Co. , holding that Congress had the power under the Commerce Clause to regulate employment conditions .
</item 4>

<item 5>
Minimum wage in the United States
20th century ( edit ) In 1910 , in conjunction with advocacy work led by Florence Kelley of the National Consumer League , the Women 's Trade Union League (
WTLU ) of Massachusetts under the leadership of Elizabeth Evans took up the cause of minimum wage legislation in Massachusetts . Over the next two years , a
coalition of social reform groups and labor advocates in Boston pushed for minimum wage legislation in the state . In 1912 , Massachusetts passed the first
American minimum wage legislation , which established a state commission for recommending non-compulsory minimum wages for women and children . Within eight
years , at least thirteen U.S. states and the District of Columbia would pass minimum wage laws , with pressure being placed on state legislatures by the
National Consumers League in a coalition with other women 's voluntary associations and organized labor . The United States Supreme Court of the Lochner era
consistently invalidated compulsory minimum wage laws . Advocates for these minimum wage laws hoped that they would be upheld under the precedent of Muller v.
Oregon , which upheld maximum working hours laws for women on the grounds that women required special protection that men did not .
</item 5>

<item 6>
Minimum wage in the United States
From January 1981 to April 1990 , the minimum wage was frozen at $3.35 per hour , then a record - setting minimum wage freeze . From September 1 , 1997 through
July 23 , 2007 , the federal minimum wage remained constant at $5.15 per hour , breaking the old record . From the United States Department of Labor .
Employment Standards Administration . Wage and Hour Division , the source page has a clickable US map with current and projected state - by - state minimum wage
rates for each state . Some government entities , such as counties and cities , observe minimum wages that are higher than the state as a whole . One notable
example of this is Santa Fe , New Mexico , whose $9.50 per hour minimum wage was the highest in the nation , until San Francisco increased its minimum wage to
$9.79 in 2009 . Another device to increase wages , living wage ordinances , generally apply only to businesses that are under contract to the local government
itself . Since 1984 , the purchasing power of the federal minimum wage has decreased .
</item 6>

<item 7>
Minimum wage in the United States
The Annals of the American Academy of Political and Social Science . 48 : 27 . Jump up ^ Broda , Rudolf ; United States . Bureau of Labor Statistics ( December
1928 ) . Minimum Wage Legislation in Various Countries : Bulletin of the United States Bureau of Labor Statistics , No. 467 ( PDF ) . Bulletin of the United
States Bureau of Labor Statistics . G.P.O. p. 11 . Jump up ^ Jerold L. , , Waltman ( 2000 ) . The politics of the minimum wage . Urbana : University of Illinois
Press . p. 11 . ISBN 9780252025457 . OCLC 42072067 . Jump up ^ Vivien. , Hart , ( 2001 ) . Bound by Our Constitution : Women , Workers , and the Minimum Wage .
Princeton : Princeton University Press . p. 63 . ISBN 9781400821563 . OCLC 700688619 . Jump up ^ Vivien. , Hart , ( 2001 ) .
</item 7>

<item 8>
Minimum wage in the United States
Bureau of Labor Statistics '' . Bls.gov . Retrieved 18 October 2017 . Jump up ^ Tritch , Teresa ( March 7 , 2014 ) . `` F.D.R. Makes the Case for the Minimum
Wage '' . New York Times . Retrieved March 7 , 2014 . Jump up ^ Broda , Rudolf ; United States . Bureau of Labor Statistics ( December 1928 ) . Minimum Wage
Legislation in Various Countries : Bulletin of the United States Bureau of Labor Statistics , No. 467 ( PDF ) . Bulletin of the United States Bureau of Labor
Statistics . G.P.O. p. 8 . Jump up ^ Willis J. , Nordlund ( 1997 ) . The quest for a living wage : the history of the federal minimum wage program . Westport ,
Conn. : Greenwood Press . p. 2 . ISBN 9780313264122 . OCLC 33983425 . Jump up ^ Hammond , Matthew B. ( 1913 ) . `` The Minimum Wage in Great Britain and
Australia '' .
</item 8>

<item 9>
Minimum wage in the United States
, holding that Congress had the power under the Commerce Clause to regulate employment conditions . The 1938 minimum wage law only applied to `` employees
engaged in interstate commerce or in the production of goods for interstate commerce , '' but in amendments in 1961 and 1966 , the federal minimum wage was
extended ( with slightly different rates ) to employees in large retail and service enterprises , local transportation and construction , state and local
government employees , as well as other smaller expansions ; a grandfather clause in 1990 drew most employees into the purview of federal minimum wage policy ,
which now set the wage at $3.80 . 21st century ( edit ) Legislation ( edit ) Play media An opinion video from the Economic Policy Institute about working for
the tipped minimum wage of $2.13 per hour . In 2006 , voters in six states ( Arizona , Colorado , Missouri , Montana , Nevada , and Ohio ) approved statewide
increases in the state minimum wage . The amounts of these increases ranged from $1 to $1.70 per hour and all increases were designed to annually index to
inflation .
</item 9>

<item 10>
Minimum wage in the United States
Federal ( edit ) Type Min . wage ( $ / h ) Notes Tipped $2.13 The Fair Labor Standards Act of 1938 has been requiring a minimum wage of $2.13 for tipped workers
with the expectation that wages plus tips total no less than $7.25 per hour since September 1 , 1991 . The employer must pay the difference if total income does
not add up to $7.25 per hour . Non-tipped $7.25 Per the Fair Minimum Wage Act of 2007 ( FMWA ) since July 24 , 2009 . Youth $4.25 The Fair Labor Standards Act
has since September 1 , 1996 allowed for persons under the age of 20 to be paid $4.25 for the first 90 calendar days of their employment . State ( edit ) The
average US minimum wage per capita ( 2017 ) is $8.49 based on the population size of each state and generally represents the average minimum wage experienced by
a person working in one of the fifty US states . Cities , counties , districts , and territories are not included in the calculation .
</item 10>
=== END ITEM id=18f2599731c9ee4a ===
