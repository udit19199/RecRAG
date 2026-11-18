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

=== ITEM id=a0c77afd422728bb ===
Gold sentences and facts:
Sentence 0: Chance 's friend Pat Wheeler ( Ward Bond ) and his wagon train of supplies stop in town , with a young gunslinger , Colorado Ryan ( Ricky Nelson ) , riding guard . Inside the jail , Stumpy ( Walter Brennan ) , Chance 's game - legged deputy , keeps watch over the jail and Joe , who knows that Stumpy holds an old grudge against Joe 's wealthy and powerful brother . Joe warns his jailers that Nathan Burdette will not like how his brother is being treated .
  Fact 0.0: Pat Wheeler is Chance's friend.
  Fact 0.1: Pat Wheeler is played by Ward Bond.
  Fact 0.2: Pat Wheeler has a wagon train of supplies.
  Fact 0.3: Pat Wheeler and his wagon train of supplies stop in town.
  Fact 0.4: Colorado Ryan is a young gunslinger.
  Fact 0.5: Colorado Ryan is played by Ricky Nelson.
  Fact 0.6: Colorado Ryan is riding guard as Pat Wheeler's wagon train stops in town.
  Fact 0.7: Joe is in the jail.
  Fact 0.8: Stumpy is played by Walter Brennan.
  Fact 0.9: Stumpy is Chance's game-legged deputy.
  Fact 0.10: Stumpy keeps watch over the jail.
  Fact 0.11: Stumpy keeps watch over Joe.
  Fact 0.12: Joe knows that Stumpy holds an old grudge against Joe's wealthy and powerful brother.
  Fact 0.13: Joe's brother is wealthy and powerful.
  Fact 0.14: Joe warns his jailers.
  Fact 0.15: Joe warns his jailers that Nathan Burdette will not like how Joe, Nathan's brother, is being treated.
Sentence 1: Walter Brennan as Stumpy
  Fact 1.0: Walter Brennan plays the character Stumpy.

Retrieved items:
<item 1>
Rio Bravo (film)
After a pair of sheer tights comes floating from a window to the street , a delighted Stumpy retrieves it , but Dude cautions him to mind his own business .
Cast ( edit ) John Wayne and Angie Dickinson in Rio Bravo John Wayne as John T. Chance Dean Martin as Dude Ricky Nelson as Colorado / Ryan Angie Dickinson as
Feathers Walter Brennan as Stumpy Ward Bond as Pat Wheeler John Russell as Nathan Burdette Pedro Gonzalez Gonzalez as Carlos Robante Estelita Rodriguez as
Consuela Robante Claude Akins as Joe Burdette Joseph Shimada as Burt Malcolm Atterbury and Harry Carey Jr. also receive a screen credit in the film 's opening ,
but their scenes were deleted from the final film . Production ( edit ) Ricky Nelson performing the song `` Get Along Home , Cindy '' in Rio Bravo Exteriors for
the film were shot at Old Tucson Studios , just outside Tucson . Filming took place in the summer of 1958 , and the movie 's credits gave 1958 as the year of
production , although the film was not released until 1959 .
</item 1>

<item 2>
Rio Bravo (film)
McCampbell , the film is about the sheriff of the town of Rio Bravo , Texas , who arrests the brother of a powerful local rancher to help his drunken deputy /
friend . With the help of a cripple and a young gunfighter , they hold off the rancher 's gang . Rio Bravo was filmed on location at Old Tucson Studios outside
Tucson , Arizona , in Technicolor . In 2014 , Rio Bravo was deemed `` culturally , historically , or aesthetically significant '' by the Library of Congress and
selected for preservation in the National Film Registry . Contents ( hide ) 1 Plot 2 Cast 3 Production 4 High Noon debate 5 Reception 6 Remakes and inspirations
6.1 Remakes 6.2 Inspirations 7 Music 8 Comic book adaption 9 See also 10 References 11 Further reading 12 External links Plot ( edit ) In the town of Rio Bravo
, Texas , sheriff 's deputy Dude ( Dean Martin ) , who has acquired the contemptuous nickname Borrachón ( pronounced ( bo.
</item 2>

<item 3>
Rio Bravo (film)
Chance is ordered to take the men to the jail to let Joe go . He is reluctant to do so , but Dude implores Chance to cooperate , saying that Stumpy is alone and
has no food or water to hold out very long . The remaining Burdette men at the hotel take Dude hostage and Nathan Burdette offers to trade him later for Joe .
Stumpy opens fire at the jail , holding off Burdette 's men . Chance and Colorado then take Joe to make the trade for Dude at a warehouse , leaving Stumpy
behind because of his bad leg . During the trade , walking in opposite directions , Dude tackles Joe and they scuffle while a gunfight erupts . Stumpy turns up
and helps even the odds . Dude overpowers Joe . The lawmen throw dynamite sticks at the warehouse where Burdette and his men are based , shooting the sticks to
explode them . After a few explosions rock the warehouse , the criminals surrender . All is quiet in town as Chance gets reacquainted with Feathers , who models
a skimpy new costume she will be wearing in her new job singing at the hotel .
</item 3>

<item 4>
Rio Bravo (film)
The Nest , a 2002 film by Florent Emilio Siri , starring Samy Naceri , Benoît Magimel , Nadia Farès , Pascal Greggory , and Sami Bouajila , is a quasi-remake of
Assault on Precinct 13 . Borrachón : A Prequel to Rio Bravo , by Kevin Cullen , 2016 . This book is a prequel to the Rio Bravo story and film and provides the
complete background leading up to the classic tale . Music ( edit ) `` My Rifle , My Pony and Me '' -- sung by Dean Martin and Ricky Nelson `` Cindy '' -- sung
by Ricky Nelson , Dean Martin and Walter Brennan `` Rio Bravo '' -- sung by Dean Martin ( end credits ) Comic book adaption ( edit ) Dell Four Color # 1018 (
June 1959 ) See also ( edit ) List of American films of 1959 John Wayne filmography El Dorado ( 1966 ) Rio Lobo ( 1970 ) References ( edit ) Jump up ^ `` 1959 :
Probable Domestic Take '' , Variety , 6 January 1960 p. 34 .
</item 4>

<item 5>
Rio Bravo (film)
Inspirations ( edit ) Feathers ' dialogue was occasionally inspired by the character of `` Slim '' ( Lauren Bacall ) in the 1944 To Have and Have Not , as when
, after the first kiss , she says : `` It 's better when two people do it , '' recalling the phrase `` It 's even better when you help ; '' and again later when
she says , `` I 'm hard to get -- you 're going to have to say you want me , '' recalling Slim 's `` I 'm hard to get , Steve -- all you have to do is ask me .
'' L'homme à l'étoile d'argent ( The Man with the Silver Star ) , a 1969 album from the French comics series Lt. Blueberry was directly inspired by Rio Bravo .
The plot is virtually the same . Blueberry plays the role of sheriff John T. Chance ; McClure , a whiskey - adoring old man , combines the roles of Dude and
Stumpy ; Dusty plays the role of Colorado ; Miss March , the teacher , plays the role of a less morally challenged Feathers ; and instead of the Burdettes , it
has the Bass brothers .
</item 5>

<item 6>
Rio Bravo (film)
Rio Bravo ( film ) - wikipedia Rio Bravo ( film ) Jump to : navigation , search For other uses , see Rio Bravo . Rio Bravo Theatrical release poster Directed by
Howard Hawks Produced by Howard Hawks Screenplay by Jules Furthman Leigh Brackett Based on `` Rio Bravo '' by B.H. McCampbell Starring John Wayne Dean Martin
Ricky Nelson Angie Dickinson Walter Brennan Ward Bond John Russell Music by Dimitri Tiomkin Cinematography Russell Harlan Edited by Folmar Blangsted Production
company Warner Bros . Distributed by Warner Bros . Release date March 18 , 1959 ( 1959 - 03 - 18 ) ( USA ) Running time 141 minutes Country United States
Language English Box office $5.2 million ( est . US / Canada rentals ) Rio Bravo is a 1959 American Western film produced and directed by Howard Hawks and
starring John Wayne , Dean Martin , Ricky Nelson , Angie Dickinson , Walter Brennan , and Ward Bond . Written by Jules Furthman and Leigh Brackett , based on
the short story `` Rio Bravo '' by B.H.
</item 6>

<item 7>
Rio Bravo (film)
( 1962 ) Man 's Favorite Sport ? ( 1964 ) Red Line 7000 ( 1965 ) El Dorado ( 1966 ) Rio Lobo ( 1970 ) VIAF : 316751659 GND : 4764446 - 1 SUDOC : 104486708 BNF :
cb145169957 ( data ) Retrieved from `` https://en.wikipedia.org/w/index.php?title=Rio_Bravo_(film)&oldid=817646117 '' Categories : 1959 films English - language
films 1950s Western ( genre ) films American Western ( genre ) films American buddy films Films scored by Dimitri Tiomkin Films about alcoholism Films directed
by Howard Hawks Films set in Texas Films shot in Arizona Films shot in California Screenplays by Leigh Brackett Warner Bros.
</item 7>

<item 8>
Rio Bravo (film)
Both remember how good with a gun Dude used to be . Chance 's friend Pat Wheeler ( Ward Bond ) and his wagon train of supplies stop in town , with a young
gunslinger , Colorado Ryan ( Ricky Nelson ) , riding guard . Inside the jail , Stumpy ( Walter Brennan ) , Chance 's game - legged deputy , keeps watch over the
jail and Joe , who knows that Stumpy holds an old grudge against Joe 's wealthy and powerful brother . Joe warns his jailers that Nathan Burdette will not like
how his brother is being treated . A mysterious woman nicknamed Feathers ( Angie Dickinson ) is in the saloon , playing poker . In the meantime , Dude and
Chance patrol the town . Hotel owner Carlos ( Pedro Gonzalez - Gonzalez ) stops the sheriff , saying Wheeler has been talking too much about Chance needing help
. In the saloon , Chance implores Wheeler to stop , as it will draw attention from the wrong people . Wheeler suggests that Colorado could be of assistance ,
but Colorado politely declines , saying he wants to `` mind my own business '' . Colorado also promises not to start any trouble without telling the sheriff
first .
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
Rio Bravo (film)
In Rio Bravo , Chance is surrounded by allies -- a deputy who is brave and good with a gun , despite recovering from alcoholism ( Dude ) , a young untried but
self - assured gunfighter ( Colorado ) , a limping `` crippled '' old man who is doggedly loyal ( Stumpy ) , a Mexican innkeeper ( Carlos ) , his wife (
Consuela ) , and an attractive young woman ( Feathers ) -- and repeatedly turns down aid from anyone he does not think is capable of helping him , though in the
final shootout they come to help him anyway . `` Who 'll turn up next ? '' Wayne asks amid the gunfire , to which Colorado replies : `` Maybe the girl with
another flower pot . '' Reception ( edit ) In the United Kingdom , Rio Bravo was not originally even reviewed for Sight & Sound ; in August 1976 , Leslie
Halliwell gave the film two out of four stars , describing it as a `` cheerfully overlong and slow - moving Western '' that was nevertheless `` very watchable
for those with time to spare '' .
</item 10>
=== END ITEM id=a0c77afd422728bb ===

=== ITEM id=def0d6aa562d37bd ===
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
Problems with concentration and memory may occur . There may also be depression or anxiety . This is part of premenstrual syndrome ( PMS ) and is estimated to
occur in 20 to 30 % of women . In 3 to 8 % it is severe . More severe symptoms of anxiety or depression may be signs of premenstrual dysphoric disorder ( PMDD )
. Rarely , in individuals who are susceptible , menstruation may be a trigger for menstrual psychosis . Extreme psychological stress can also result in periods
stopping . Bleeding The average volume of menstrual fluid during a monthly menstrual period is 35 milliliters ( 2.4 tablespoons of menstrual fluid ) with 10 --
80 milliliters ( 1 -- 6 tablespoons of menstrual fluid ) considered typical . Menstrual fluid is the correct name for the flow , although many people prefer to
refer to it as menstrual blood . Menstrual fluid contains some blood , as well as cervical mucus , vaginal secretions , and endometrial tissue . Menstrual fluid
is reddish - brown , a slightly darker color than venous blood . About half of menstrual fluid is blood .
</item 1>

<item 2>
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
</item 2>

<item 3>
Menstruation
Scotland is the first country to ban period poverty . In 2015 a petition was put forward in England seeking the elimination of the 5 % `` tampon tax '' . The
government voted in favour but it has yet to be implemented due to Brexit complications . In England , one in ten women between 14 and 21 can not afford
menstrual management products . In the United States , nine states have eliminated the tampon tax , and seven states have introduced legislation . In January
2018 , California rejected a proposal to eliminate tampon tax . Evolution All female placental mammals have a uterine lining that builds up when the animal is
fertile , but it is dismantled when the animal is infertile . Most female mammals have an estrous cycle , yet only primates ( including humans ) , several
species of bats , and elephant shrews have a menstrual cycle . ( Killer whales , Orcinus orca , are one of the few species to experience menopause . ) Some
anthropologists have questioned the energy cost of rebuilding the endometrium every fertility cycle .
</item 3>

<item 4>
Menstruation
Menstrual cups are usually made of silicone and can last 5 years or longer . Reusable cloth pads -- Pads that are made of cotton ( often organic ) , terrycloth
, or flannel , and may be handsewn ( from material or reused old clothes and towels ) or storebought . Padded panties or period - proof underwear -- Reusable
cloth ( usually cotton ) underwear with extra absorbent layers sewn in to absorb flow . Some also use patented technology to be leak resistant , such as the
brand THINX . Sea sponges -- Natural sponges , worn internally like a tampon to absorb menstrual flow . Blanket , towel -- ( also known as a draw sheet ) --
large reusable piece of cloth , most often used at night , placed between legs to absorb menstrual flow . Non-commercial items In developing countries , many
poor women can not afford commercial feminine hygiene products . Instead , they use materials found in the environment or other improvised materials .
</item 4>

<item 5>
Menstruation
Some anthropologists have questioned the energy cost of rebuilding the endometrium every fertility cycle . However , anthropologist Beverly Strassmann has
proposed that the energy savings of not having to continuously maintain the uterine lining more than offsets energy cost of having to rebuild the lining in the
next fertility cycle , even in species such as humans where much of the lining is lost through bleeding ( overt menstruation ) rather than reabsorbed ( covert
menstruation ) . Many have questioned the evolution of overt menstruation in humans and related species , speculating on what advantage there could be to losing
blood associated with dismantling the endometrium , rather than absorbing it , as most mammals do . Humans do , in fact , reabsorb about two - thirds of the
endometrium each cycle . Strassmann asserts that overt menstruation occurs not because it is beneficial in itself . Rather , the fetal development of these
species requires a more developed endometrium , one which is too thick to reabsorb completely . Strassman correlates species that have overt menstruation to
those that have a large uterus relative to the adult female body size .
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
Instead , they use materials found in the environment or other improvised materials . Absorption materials that may be used by women who can not afford anything
else include : sand , ash , small hole in earth , cloth , whole leaf , leaf fibre ( such as water hyacinth , banana , papyrus , cotton fibre ) , paper ( toilet
paper , re-used newspaper , pulped and dried paper ) , animal pelt e.g. goat skin , double layer of underwear , skirt or sari . Society and culture Amra Padatik
India , celebration of Menstrual Hygiene Day in India Traditions and taboos Main article : Culture and menstruation Further information : Menstrual taboo Many
religions have menstruation - related traditions . Islam Islam prohibits sexual contact with women during menstruation in the 2nd chapter of the Quran . Women
are exempted from Salah ( prayer ) . They are also prohibited from fasting and obligatory fasts of Ramadan are to be performed during other days . During menses
pilgrimages are allowed ; however , circumambulation of the Kaaba is prohibited and is to be performed during other times .
</item 7>

<item 8>
Menstruation
Vaginal pH is higher and thus less acidic than normal , the cervix is lower in its position , the cervical opening is more dilated , and the uterine endometrial
lining is absent , thus allowing organisms direct access to the bloodstream through the numerous blood vessels that nourish the uterus . All these conditions
increase the chance of infection during menstruation . Education Menstruation education is frequently taught in combination with sex education in the US ,
although one study suggests that girls would prefer their mothers to be the primary source of information about menstruation and puberty . A Nigerian study
showed the following breakdown in menstruation education : `` parents of 56.1 % , friends of 53.18 % , books of 45.56 % , teachers of 44.15 % , internet of
45.19 % , and health centers of 54.14 % '' held the most influence in terms of menstruation education . Information about menstruation is often shared among
friends and peers , which may promote a more positive outlook on puberty . The quality of menstrual education in a society determines the accuracy of people 's
understanding of the process .
</item 8>

<item 9>
Menstruation
Menstruation - wikipedia Menstruation Jump to : navigation , search This article is about humans . For other mammals , see Menstruation ( mammal ) . Figure
showing the progression of the menstrual cycle and the different hormones contributing to it . Menstruation , also known as a period or monthly , is the regular
discharge of blood and mucosal tissue ( known as menses ) from the inner lining of the uterus through the vagina . The first period usually begins between
twelve and fifteen years of age , a point in time known as menarche . However , periods may occasionally start as young as eight years old and still be
considered normal . The average age of the first period is generally later in the developing world , and earlier in the developed world . The typical length of
time between the first day of one period and the first day of the next is 21 to 45 days in young women , and 21 to 31 days in adults ( an average of 28 days ) .
Bleeding usually lasts around 2 to 7 days . Menstruation stops occurring after menopause , which usually occurs between 45 and 55 years of age .
</item 9>

<item 10>
Menstruation
sex Life span Prenatal development / Sexual dimorphism / Sexual differentiation ( Feminization Virilization ) Puberty ( Gonadarche Pubarche Menarche Spermarche
Adrenarche ) Maternal age / Paternal age Climacteric ( Menopause Late - onset hypogonadism ) Tanner scale Egg Ovum Oviposition Oviparity Ovoviviparity Vivipary
Reproductive endocrinology and infertility Hypothalamic -- pituitary -- gonadal axis Hypothalamic -- pituitary -- prolactin axis Andrology Hormone Breast
Thelarche Breast development Lactation Breastfeeding Menstrual cycle Events and phases Menstruation Follicular phase Ovulation Luteal phase Life stages Menarche
Menopause Tracking Signs Basal body temperature Cervical mucus Mittelschmerz Systems Fertility awareness Calendar - based methods Billings Ovulation Method
Creighton Model Suppression Extended cycle combined hormonal contraceptive Lactational amenorrhea Disorders Amenorrhea Anovulation Dysmenorrhea Hypomenorrhea
Irregular menstruation Menometrorrhagia Menorrhagia Metrorrhagia Oligomenorrhea Related events
</item 10>
=== END ITEM id=def0d6aa562d37bd ===

=== ITEM id=4969ed5d90c8eebc ===
Gold sentences and facts:
Sentence 0: The Raiders had a great chance to score a touchdown early in the game after cornerback Charles Woodson intercepted Buccaneers quarterback Brad Johnson 's pass on the third play of the game and returned it 12 yards to the Tampa Bay 36 - yard line . However , six plays later , Tampa Bay defensive end Simeon Rice sacked Raiders quarterback Rich Gannon on third down , forcing Oakland to settle for kicker Sebastian Janikowski 's 40 - yard field goal to give them a 3 -- 0 lead .
  Fact 0.0: Cornerback Charles Woodson intercepted a pass thrown by Buccaneers quarterback Brad Johnson.
  Fact 0.1: The interception happened on the third play of the game.
  Fact 0.2: Charles Woodson returned the interception 12 yards to the Tampa Bay 36-yard line.
  Fact 0.3: The Raiders had a great chance to score a touchdown early in the game after the interception.
  Fact 0.4: Six plays after the interception, Tampa Bay defensive end Simeon Rice sacked Raiders quarterback Rich Gannon on third down.
  Fact 0.5: The sack forced Oakland to settle for a field goal.
  Fact 0.6: Kicker Sebastian Janikowski kicked a 40-yard field goal for Oakland.
  Fact 0.7: The field goal gave Oakland a 3-0 lead.
Sentence 1: Still , even Gruden had trouble getting the offense in sync during his first year as Tampa Bay 's head coach . In 2002 , the Buccaneers ranked 25th in the league in total yards gained ( 5,222 ) . Quarterback Brad Johnson made the Pro Bowl , completing 281 out of 451 passes for 3,049 yards , 22 touchdowns , and only 6 interceptions . Running back Michael Pittman led the team in rushing with 718 yards and one touchdown , and caught 59 passes for 477 yards . Pro Bowl fullback Mike Alstott had 548 rushing yards and 5 touchdowns , and also had 35 receptions for 242 yards and 2 touchdowns . Wide receiver Keyshawn Johnson led the team with 76 receptions for 1,088 yards and 5 touchdowns , while wide receiver Keenan McCardell had 61 receptions for 670 yards and 6 touchdowns .
  Fact 1.0: Even Gruden had trouble getting the offense in sync during his first year as Tampa Bay's head coach.
  Fact 1.1: In 2002, the Buccaneers ranked 25th in the league in total yards gained.
  Fact 1.2: The Buccaneers gained 5,222 total yards in 2002.
  Fact 1.3: Quarterback Brad Johnson made the Pro Bowl.
  Fact 1.4: Brad Johnson completed 281 of 451 passes for 3,049 yards.
  Fact 1.5: Brad Johnson threw 22 touchdowns.
  Fact 1.6: Brad Johnson threw only 6 interceptions.
  Fact 1.7: Running back Michael Pittman led the Buccaneers in rushing with 718 yards and one touchdown.
  Fact 1.8: Michael Pittman caught 59 passes for 477 yards.
  Fact 1.9: Pro Bowl fullback Mike Alstott had 548 rushing yards and 5 touchdowns.
  Fact 1.10: Mike Alstott had 35 receptions for 242 yards and 2 touchdowns.
  Fact 1.11: Wide receiver Keyshawn Johnson led the team with 76 receptions for 1,088 yards and 5 touchdowns.
  Fact 1.12: Wide receiver Keenan McCardell had 61 receptions for 670 yards and 6 touchdowns.

Retrieved items:
<item 1>
Super Bowl XXXVII
Then on third and 18 from the Oakland 29 - yard line , Tampa Bay linebacker Derrick Brooks intercepted a pass from Gannon and returned it 44 yards for a
touchdown , giving the Buccaneers a 41 -- 21 lead with only 1 : 18 left , and leading Buccaneers radio announcer Gene Deckerhoff to make his famous call of ``
The dagger 's in , we 're gon na win the Super Bowl ! '' . A few plays later , with the Raiders now playing for pride , Dwight Smith intercepted a tipped pass
and returned it 50 yards for a touchdown with only two seconds remaining . Gramatica kicked the extra point , and the Buccaneers were up 48 -- 21 . Raiders
defensive tackle Chris Cooper returned Gramatica 's kickoff 6 yards before being tackled by Jack Golden , ending the game . The Buccaneers won their first -
ever Super Bowl with a 48 - 21 victory . Gannon said after the game that his performance was `` nightmarish '' .
</item 1>

<item 2>
Super Bowl XXXVII
Super Bowl XXXVII - wikipedia Super Bowl XXXVII Jump to : navigation , search Super Bowl XXXVII Oakland Raiders ( AFC ) Tampa Bay Buccaneers ( NFC ) 21 48 Total
OAK 0 6 12 21 TB 17 14 14 48 Date January 26 , 2003 ( 2003 - 01 - 26 ) Stadium Qualcomm Stadium , San Diego , California MVP Dexter Jackson , Safety Favorite
Raiders by 4 Referee Bill Carollo Attendance 67,603 Future Hall of Famers Buccaneers : Derrick Brooks , Warren Sapp Raiders : Al Davis ( owner / general manager
) , Tim Brown , Jerry Rice , Rod Woodson Ceremonies National anthem Dixie Chicks Coin toss 1972 Miami Dolphins : Don Shula , Bob Griese , Larry Csonka , Larry
Little , Jim Langer , Nick Buoniconti , Paul Warfield Halftime show Shania Twain , No Doubt , and Sting TV in the United States Network ABC Announcers Al
Michaels , John Madden , Melissa Stark and Lynn Swann Nielsen ratings 40.7 ( est .
</item 2>

<item 3>
Super Bowl XXXVII
South Florida Sun - Sentinel . Retrieved August 27 , 2012 . ^ Jump up to : `` Bucs stop here -- with a Super Bowl title '' . USA Today . January 26 , 2003 .
Retrieved August 27 , 2012 . Jump up ^ Pompei , Dan ( January 27 , 2003 ) . `` The best - laid plans : an inside look at how coach Jon Gruden and the Bucs
prepared themselves to be Super Bowl champions '' . The Sporting News . Archived from the original on August 19 , 2003 . Jump up ^ `` Super Bowl XXXVII --
National Football League Game Summary '' ( PDF ) . National Football League . January 26 , 2003 . Retrieved January 8 , 2017 . Jump up ^ Martin , Glen ; Asimov
, Nanette ; et al. ( January 27 , 2003 ) . `` Raider Rage : Oakland police no match for street mayhem '' . San Francisco Chronicle . p . A1 . Archived from the
original on February 10 , 2003 .
</item 3>

<item 4>
Super Bowl XXXVII
This was the last Super Bowl until Super Bowl 50 to take place in California . It was the first and only time until Super Bowl XLVIII in Super Bowl history that
the league 's number one - ranked offense ( Raiders ) was pitted against the league 's number one - ranked defense ( Buccaneers ) . Super Bowl XXXVII is also
referred to as the `` Pirate Bowl '' , due to both teams ' pirate - themed names . The game is sometimes referred to as the `` Gruden Bowl '' , because the
primary storyline surrounding the game revolved around Jon Gruden . Gruden was the Raiders ' head coach from 1998 to 2001 , and then became the Buccaneers coach
in 2002 . Tampa Bay , `` Gruden 's new team '' , made their first Super Bowl appearance in team history after posting a 12 -- 4 regular season record . Oakland
, `` Gruden 's old team '' , advanced to their fifth Super Bowl after an 11 -- 5 regular season . This was also the first Super Bowl to feature two former
division rivals , as the Buccaneers and Raiders both played in the AFC West in 1976 .
</item 4>

<item 5>
Super Bowl XXXVII
Tampa Bay responded by moving the ball to the Oakland 9 - yard line on their ensuing drive , featuring a 24 - yard run by Pittman , but they came up empty after
Tupa fumbled the snap on a field goal attempt . A few plays later , Gannon threw a 48 - yard touchdown pass to wide receiver Jerry Rice with 6 : 06 left in the
game , cutting the Raiders deficit to 34 -- 21 . The two - point conversion failed when Jerry Porter caught the ball but landed out of bounds . Though there was
contact with a defender , the officials deemed the contact incidental rather than a force - out , and therefore that part of the play was non-reviewable . In an
attempt to prevent a Raiders comeback , the Buccaneers managed to run the clock down to 2 : 44 on their ensuing drive before being forced to punt .
</item 5>

<item 6>
Super Bowl XXXVII
Gurley 91 Chuck Darby 92 Anthony McFarland 93 DeVone Claybrooks 94 Greg Spires 95 Ron Warner 96 Ellis Wyms 97 Simeon Rice 98 Corey Smith 99 Warren Sapp -- Onome
Ojo -- T.T. Toliver Head coach : Jon Gruden Coaches : Joe Barry Jeremy Bates Richard Bisaccia Michael Christianson Jay Gruden Monte Kiffin Richard Mann Rod
Marinelli Raheem Morris Bill Muir Johnny Parker Stan Parrish Mike Tomlin Art Valero Kirby Wilson Tampa Bay Buccaneers Founded in 1976 Based and headquartered in
Tampa , Florida Franchise Franchise Seasons History Coaches Players Expansion draft First - round draft picks Draft history Starting quarterbacks Broadcasters
Stadiums Tampa Stadium Raymond James Stadium Culture and lore Hugh Culverhouse Malcolm Glazer Tampa 2 Captain Fear Cheerleaders Snow Bowl A Triumph of the Heart
: The Ricky Bell Story Rivalries Carolina Panthers New Orleans Saints Miami Dolphins Playoff appearances ( 10 ) 1979 1981 1982 1997 1999 2000 2001 2002 2005
2007 Division championships ( 6 ) 1979 1981 1999
</item 6>

<item 7>
Super Bowl XXXVII
The Raiders came into the game as four - point favorites . However , the Tampa Bay defense dominated the contest . Oakland quarterback Rich Gannon threw a Super
Bowl record five interceptions , three of which were returned for touchdowns . The Buccaneers also sacked Gannon five times , and scored 34 consecutive points
to build a 34 -- 3 lead late in the third quarter . Tampa Bay safety Dexter Jackson , who had two of those interceptions and returned them for 34 yards , was
named Super Bowl MVP. Jackson became only the second safety and third defensive back named Super Bowl MVP .
</item 7>

<item 8>
Super Bowl XXXVII
January 15 , 2003 . Retrieved August 27 , 2012 . Jump up ^ Mitchell , Fred ( January 27 , 2003 ) . `` In their fashion , Bucs fans laugh '' . Chicago Tribune .
Retrieved August 27 , 2012 . Jump up ^ Vivinetto , Gina ( January 27 , 2003 ) . `` Super Bowl XXXVII : Superstars belt out favorites , or lip synch them '' .
St. Petersburg Times . ^ Jump up to : George , Thomas ( January 27 , 2003 ) . `` Bucs steal the treasure '' . The New York Times . Retrieved August 27 , 2012 .
^ Jump up to : `` Tampa Bay Buccaneers 48 vs. Oakland Raiders 21 '' . pro-football - reference . Retrieved August 27 , 2012 . Jump up ^ Cunningham , Michael ;
Skolnick , Ethan J. ( January 27 , 2003 ) . `` Treu Gets The Call In Robbins ' Absence '' . South Florida Sun - Sentinel .
</item 8>

<item 9>
Super Bowl XXXVII
88.6 million viewers ) Market share 61 Cost of 30 - second commercial $2.1 million ← XXXVI Super Bowl XXXVIII → Super Bowl XXXVII was an American football game
between the American Football Conference ( AFC ) champion Oakland Raiders and the National Football Conference ( NFC ) champion Tampa Bay Buccaneers to decide
the National Football League ( NFL ) champion for the 2002 season . The Buccaneers defeated the Raiders by the score of 48 -- 21 , tied with Super Bowl XXXV for
the seventh largest Super Bowl margin of victory , and winning their first ever Super Bowl . The game , played on January 26 , 2003 at Qualcomm Stadium in San
Diego , California , was the sixth Super Bowl to be held a week after the conference championship games ( XVII , XXV , XXVIII , XXXIV , and XXXVI ) . It was
also the last Super Bowl played in the month of January . Super Bowl XXXVI was the first to be played in February , due to the NFL postponing games for a week
after the September 11 attacks . Starting with Super Bowl XXXVIII in 2004 , the Super Bowl has been permanently played in February .
</item 9>

<item 10>
Graph query row:
person: Brad Johnson
event: Super Bowl XXXVII
team: Tampa Bay Buccaneers
participation: PARTICIPATED_IN
result: None
outcome: None
team_role: None
</item 10>
=== END ITEM id=4969ed5d90c8eebc ===

=== ITEM id=bf62c0f59ed5655d ===
Gold sentences and facts:
Sentence 0: Sport Men Women Total Cross-country skiing 0 Total 0
  Fact 0.0: The table has columns for Sport, Men, Women and Total.
  Fact 0.1: The table lists Cross-country skiing as a sport.
  Fact 0.2: The table lists a total of 0 for Cross-country skiing.
  Fact 0.3: The table lists a grand total of 0.
Sentence 1: Bermuda competed at the 2018 Winter Olympics in Pyeongchang , South Korea , from 9 to 25 February 2018 , with one competitor . This was the nation sixth appearance in the Winter Olympics .
  Fact 1.0: Bermuda competed at the 2018 Winter Olympics.
  Fact 1.1: The 2018 Winter Olympics were held in Pyeongchang, South Korea.
  Fact 1.2: Bermuda competed from 9 to 25 February 2018.
  Fact 1.3: Bermuda had one competitor at the 2018 Winter Olympics.
  Fact 1.4: The 2018 Winter Olympics was Bermuda's sixth appearance in the Winter Olympics.

Retrieved items:
<item 1>
Bermuda at the 2018 Winter Olympics
12 December 2017 . Retrieved 16 December 2017 . Jump up ^ `` Tucker Murphy Competes In Skiing In Iceland '' . Bernews. Hamilton , Bermuda . 14 May 2017 .
Retrieved 16 December 2017 .
</item 1>

<item 2>
Bermuda at the 2018 Winter Olympics
Sport Men Women Total Cross-country skiing 0 Total 0 Cross-country Skiing ( edit ) Main articles : Cross-country skiing at the 2018 Winter Olympics and
Cross-country skiing at the 2018 Winter Olympics -- Qualification Bermuda qualified one male athlete , Tucker Murphy . Murphy will be making his third straight
appearance at the Winter Olympics . Distance Athlete Event Final Time Deficit Rank Tucker Murphy Men 's 15 km freestyle 43 : 05.7 + 9 : 21.8 104 See also ( edit
) Bermuda at the 2018 Commonwealth Games References ( edit ) Jump up ^ `` Opening Ceremony Flagbearers - Olympic Winter Games , PyeongChang 2018 '' ( PDF ) .
olympic.org . International Olympic Committee ( IOC ) . 9 February 2018 . Retrieved 10 February 2018 . Jump up ^ `` Cross-country Skiing Quota List for Olympic
Games 2018 '' . www.data.fis-ski.com/ . International Ski Federation ( FIS ) . 12 December 2017 . Retrieved 16 December 2017 .
</item 2>

<item 3>
2018 Winter Olympics
Moritz 1952 Oslo 1956 Cortina d'Ampezzo 1960 Squaw Valley 1964 Innsbruck 1968 Grenoble 1972 Sapporo 1976 Innsbruck 1980 Lake Placid 1984 Sarajevo 1988 Calgary
1992 Albertville 1994 Lillehammer 1998 Nagano 2002 Salt Lake City 2006 Turin 2010 Vancouver 2014 Sochi 2018 Pyeongchang 2022 Beijing 2026 TBD 2030 TBD Ancient
Olympic Games Intercalated Games 1906 Paralympic Games Youth Olympic Games Events at the 2018 Winter Olympics ( PyeongChang , South Korea ) Alpine skiing
Biathlon Bobsleigh Cross-country skiing Curling Figure skating Freestyle skiing Ice hockey men women Luge Nordic combined Short track speed skating Skeleton Ski
jumping Snowboarding Speed skating Nations at the 2018 Winter Olympics in Pyeongchang , South Korea Africa Eritrea Kenya Madagascar Morocco South Africa Togo
Americas Argentina Bermuda Bolivia Brazil Canada Chile Colombia Dominica Ecuador Mexico Peru Puerto Rico United States Asia China Hong Kong India Iran Japan
Kazakhstan Kyrgyzstan Lebanon Malaysia Mongolia
</item 3>

<item 4>
2018 Winter Olympics
Mexico Peru Puerto Rico United States Asia China Hong Kong India Iran Japan Kazakhstan Kyrgyzstan Lebanon Malaysia Mongolia North Korea Pakistan Philippines
Singapore South Korea Chinese Taipei Thailand Timor - Leste Uzbekistan Europe Albania Andorra Armenia Austria Azerbaijan Belarus Belgium Bosnia and Herzegovina
Bulgaria Croatia Cyprus Czech Republic Denmark Estonia Finland France Georgia Germany Great Britain Greece Hungary Iceland Ireland Israel Italy Kosovo Latvia
Liechtenstein Lithuania Luxembourg Macedonia Malta Moldova Monaco Montenegro Netherlands Norway Poland Portugal Romania San Marino Serbia Slovakia Slovenia
Spain Sweden Switzerland Turkey Ukraine Oceania Australia New Zealand Other Olympic Athletes from Russia Venues of the 2018 Winter Olympics Pyeongchang Mountain
Cluster Alpensia Biathlon Centre Cross-Country Centre Ski Jumping Centre Convention Centre IBC Alpensia Sliding Centre Yongpyong Alpine Centre Pyeongchang
Olympic Stadium Pyeongchang Olympic Village Bokwang Phoenix Park Stand - alone venues : Jeongseon Alpine Centre Phoenix Snow Park Gangneung Coastal Cluster
Gangneung Olympic Park Gangneung Curling Centre Gangneung Hockey Centre Gangneung Ice Arena Gangneung Oval Kwandong Hockey Centre Gangneung Olympic Village
Italics indicate non-competition venues .
</item 4>

<item 5>
Beijing
Cancelled due to World War II Summer Paralympic Games host cities 1960 : Rome 1964 : Tokyo 1968 : Tel Aviv 1972 : Heidelberg 1976 : Toronto 1980 : Arnhem 1984 :
New York City / Stoke Mandeville 1988 : Seoul 1992 : Barcelona / Madrid 1996 : Atlanta 2000 : Sydney 2004 : Athens 2008 : Beijing 2012 : London 2016 : Rio de
Janeiro 2020 : Tokyo 2024 : Paris 2028 : Los Angeles Winter Olympic Games host cities 1924 : Chamonix 1928 : St. Moritz 1932 : Lake Placid 1936 : Garmisch -
Partenkirchen 1940 : Cancelled due to World War II 1944 : Cancelled due to World War II 1948 : St. Moritz 1952 : Oslo 1956 : Cortina d'Ampezzo 1960 : Squaw
Valley 1964 : Innsbruck 1968 : Grenoble 1972 : Sapporo 1976 : Innsbruck 1980 : Lake Placid
</item 5>

<item 6>
2018 Winter Olympics
North Korean athletes will be allowed to cross the DMZ into South Korea since two North Korean figure skaters qualified . North Korea agreed in negotiations
with South Korea to send the qualified athletes for the Winter Olympics . Russia 's status remained unclear due to the state - sponsored doping scandal . On 5
December 2017 it was announced that the Russian Olympic Committee was suspended effective immediately . Individual athletes who qualified and can demonstrate
they have complied with the IOC 's doping regulations will be allowed to compete as `` Olympic Athletes from Russia , '' ( OAR ) under a neutral flag and with
the Olympic anthem played in any ceremony . ( hide ) Participating National Olympic Committees Albania Andorra Argentina Armenia Australia Austria Azerbaijan
Belarus Belgium Bermuda Bolivia Bosnia and Herzegovina Brazil Bulgaria Canada Chile China Colombia Croatia Cyprus Czech Republic Denmark Dominica Ecuador
Eritrea Estonia Finland France Georgia Germany Great Britain Greece Hong Kong Hungary Iceland India Iran Ireland Israel Italy Japan Kazakhstan Kenya Kosovo
Kyrgyzstan Latvia Lebanon Liechtenstein Lithuania Luxembourg Macedonia Madagascar Malaysia Malta Mexico Moldova Monaco Mongolia Montenegro Morocco Netherlands
New Zealand North Korea Norway Olympic Athletes from Russia Pakistan Peru Philippines Poland Portugal Puerto Rico Romania San Marino Serbia Singapore Slovakia
Slovenia South Africa South Korea ( host ) Spain Sweden
</item 6>

<item 7>
2018 Winter Olympics
Numbers in parentheses indicate the number of medal events contested in each sports discipline . Alpine skiing ( 11 ) ( details ) Biathlon ( 11 ) ( details )
Bobsleigh ( 3 ) ( details ) Cross-country skiing ( 12 ) ( details ) Curling ( 3 ) ( details ) Figure skating ( 5 ) ( details ) Freestyle skiing ( 10 ) ( details
) Ice hockey ( 2 ) ( details ) Luge ( 4 ) ( details ) Nordic combined ( 3 ) ( details ) Short track speed skating ( 8 ) ( details ) Skeleton ( 2 ) ( details )
Ski jumping ( 4 ) ( details ) Snowboarding ( 10 ) ( details ) Speed skating ( 14 ) ( details ) Participating national Olympic committees ( edit ) A total of 90
teams have qualified at least one athlete so far ( 89 nations , and the delegation from Russia which will compete under the IOC flag ) . Ecuador , Eritrea ,
Kosovo , Malaysia , and Singapore are scheduled to make their Winter Olympics debut if they decide to participate .
</item 7>

<item 8>
Bermuda at the 2018 Winter Olympics
Retrieved from `` https://en.wikipedia.org/w/index.php?title=Bermuda_at_the_2018_Winter_Olympics&oldid=827912272 '' Categories : Nations at the 2018 Winter
Olympics Bermuda at the Winter Olympics by year 2018 in Bermuda Hidden categories : Use dmy dates from February 2018 Use British English from February 2018 Talk
Contents About Wikipedia Azərbaycanca Dansk Deutsch Eesti Español Français 한국어 हिन्दी Italiano Nederlands 日本 語 Norsk Polski Português Русский Suomi Svenska
Українська 中文 10 more Edit links This page was last edited on 27 February 2018 , at 13 : 44 . Text is available under the Creative Commons Attribution -
ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark
of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 8>

<item 9>
Bermuda at the 2018 Winter Olympics
14 May 2017 . Retrieved 16 December 2017 . Nations at the 2018 Winter Olympics in Pyeongchang , South Korea Africa Eritrea Ghana Kenya Madagascar Morocco
Nigeria South Africa Togo Americas Argentina Bermuda Bolivia Brazil Canada Chile Colombia Ecuador Jamaica Mexico Puerto Rico United States Asia China East Timor
Hong Kong India Iran Japan Kazakhstan Kyrgyzstan Lebanon Malaysia Mongolia North Korea Pakistan Philippines Singapore South Korea Chinese Taipei Thailand
Uzbekistan Europe Albania Andorra Armenia Austria Azerbaijan Belarus Belgium Bosnia and Herzegovina Bulgaria Croatia Cyprus Czech Republic Denmark Estonia
Finland France Georgia Germany Great Britain Greece Hungary Iceland Ireland Israel Italy Kosovo Latvia Liechtenstein Lithuania Luxembourg Macedonia Malta
Moldova Monaco Montenegro Netherlands Norway Poland Portugal Romania San Marino Serbia Slovakia Slovenia Spain Sweden Switzerland Turkey Ukraine Oceania
Australia New Zealand Tonga Other Korea Olympic Athletes from Russia Bermuda at the Olympics Summer Olympic Games 1936 1948 1952 1956 1960 1964 1968 1972 1976
1980 1984 1992 2000 2008 2012 2016 Winter Olympic Games 1992 1994 1998 2002 2006 2014 2018 Bermuda did not participate in 1980 due to a boycott .
</item 9>

<item 10>
Bermuda at the 2018 Winter Olympics
Bermuda at the 2018 Winter Olympics - wikipedia Bermuda at the 2018 Winter Olympics Jump to : navigation , search Bermuda at the 2018 Winter Olympics Flag of
Bermuda IOC code BER NOC Bermuda Olympic Association Website www.olympics.bm in Pyeongchang , South Korea 9 -- 25 February 2018 Competitors 1 in 1 sport Flag
bearer Tucker Murphy Medals Gold 0 Silver 0 Bronze 0 Total 0 Winter Olympics appearances ( overview ) 1992 1994 1998 2002 2006 2014 2018 Bermuda competed at the
2018 Winter Olympics in Pyeongchang , South Korea , from 9 to 25 February 2018 , with one competitor . This was the nation sixth appearance in the Winter
Olympics . Contents ( hide ) 1 Competitors 2 Cross-country skiing 3 See also 4 References Competitors ( edit ) The following is the list of number of
competitors participating in the delegation per sport .
</item 10>
=== END ITEM id=bf62c0f59ed5655d ===

=== ITEM id=6a9ceab1f7ac14ab ===
Gold sentences and facts:
Sentence 0: `` Too Much Time on My Hands '' is the second single released from Styx 's 1981 triple - platinum album Paradise Theatre . It was written and sung by Tommy Shaw , who also plays the lead guitar solo during the break in the song .
  Fact 0.0: 'Too Much Time on My Hands' is the second single released from Styx's 1981 triple-platinum album Paradise Theatre.
  Fact 0.1: 'Too Much Time on My Hands' was written by Tommy Shaw.
  Fact 0.2: 'Too Much Time on My Hands' was sung by Tommy Shaw.
  Fact 0.3: Tommy Shaw plays the lead guitar solo during the break in 'Too Much Time on My Hands'.

Retrieved items:
<item 1>
Closing Time (album)
`` Closing Time '' ( instrumental ) 4 : 20 Total length : 45 : 46 Personnel ( edit ) Musicians ( edit ) Tom Waits -- vocals , piano , guitar , harmonium ,
harpsichord , celeste Delbert Bennett -- trumpet Shep Cooke -- guitar , backing vocals on `` Old Shoes ( & Picture Postcards ) '' Peter Klimes -- guitar , pedal
steel guitar on `` Rosie '' Bill Plummer -- double bass John Seiter -- drums ; backing vocals on `` Ol ' 55 '' and `` Rosie '' Guest musicians ( edit ) Arni
Egilsson -- bass guitar on `` Closing Time '' ( Instrumental ) Jesse Ehrlich -- cello on `` Martha '' Tony Terran -- trumpet on `` Closing Time '' (
Instrumental ) Technical personnel ( edit ) Ed Caraeff -- photography Richie Moore -- additional engineering Cal Schenkel -- design , artwork Jerry Yester --
production , engineering Chart positions ( edit ) Chart ( 2000 ) Peak position Irish Albums Chart 29 Certifications ( edit ) Region Certification Certified
units / Sales United Kingdom ( BPI ) Gold 100,000 Summaries shipments figures based on certification alone References ( edit ) Jump up ^ `` Releases '' .
</item 1>

<item 2>
Closing Time (album)
The songs on Closing Time are often noted for their lyrical content and vary in form . `` Ol ' ' 55 '' narrates the story of a man riding `` lickety splitly ''
in a car and is often seen as a song about escapism , `` like his near - contemporary Bruce Springsteen . '' This theme is also present in `` Old Shoes '' which
narrates another story about `` a footloose young stud hitting the road and semi-sneering '' , particularly in lyrics such as `` your tears can not bind me
anymore '' and `` my heart was not born to be tamed . '' Other lyrics on the album are described as melancholic , particularly `` Lonely '' , `` I Hope That I
Do n't Fall in Love with You '' and `` Grapefruit Moon '' , which are `` both self - conscious and lacklustre . '' The intro to `` Midnight Lullaby '' borrows
lyrics from the English nursery rhyme `` Sing a Song of Sixpence '' . This form of song - writing became a lifelong habit of Waits following the writing of ``
Midnight Lullaby '' , in which he `` assembled lyrics from fragments of oral tradition . ''
</item 2>

<item 3>
Too Much Time on My Hands
Too Much Time on My Hands - Wikipedia Too Much Time on My Hands Jump to : navigation , search `` Too Much Time on My Hands '' Single by Styx from the album
Paradise Theatre B - side `` Queen of Spades '' Released March 1981 Format 7 '' Recorded 1980 Length 4 : 31 Label A&M Songwriter ( s ) Tommy Shaw Producer ( s )
Styx Styx singles chronology `` The Best of Times '' ( 1981 ) `` Too Much Time on My Hands '' ( 1981 ) `` Nothing Ever Goes as Planned '' ( 1981 ) `` The Best
of Times '' ( 1981 ) `` Too Much Time on My Hands '' ( 1981 ) `` Nothing Ever Goes as Planned '' ( 1981 ) `` Too Much Time on My Hands '' is the second single
released from Styx 's 1981 triple - platinum album Paradise Theatre . It was written and sung by Tommy Shaw , who also plays the lead guitar solo during the
break in the song . It reached No. 9 on the US Billboard Hot 100 , No. 2 on the Top Rock Tracks chart , and No.
</item 3>

<item 4>
Too Much Time on My Hands
Louis ( 2002 ) 21st Century Live ( 2003 ) One with Everything : Styx and the Contemporary Youth Orchestra ( 2006 ) The Grand Illusion / Pieces of Eight - Live (
2012 ) Live at the Orleans Arena , Las Vegas ( 2015 ) Compilations Best of Styx ( 1977 ) Lady ( 1980 ) Styx Classics Volume 15 ( 1987 ) Styx Radio - Made Hits
1975 - 1991 ( 1991 ) Greatest Hits ( 1992 ) Greatest Hits ( 1995 ) Styx Greatest Hits Part 2 ( 1996 ) The Best of Times : The Best of Styx ( 1997 ) Best of Styx
1973 - 1974 ( 1999 ) Extended Versions ( 2000 ) Singles Collection ( 2000 ) Styx Yesterday & Today ( 2001 ) 20th Century Masters ( 2002 ) Rockers ( 2003 ) Come
Sail Away -- The Styx Anthology ( 2004 ) The Complete Wooden Nickel Recordings ( 2005 ) Regeneration : Volume I & II ( 2011 & 2012 ) Cover albums
</item 4>

<item 5>
Too Much Time on My Hands
Styx . External links ( edit ) Songfacts ( hide ) Styx Chuck Panozzo James `` J.Y. '' Young Tommy Shaw Todd Sucherman Lawrence Gowan Ricky Phillips Dennis
DeYoung John Panozzo John `` J.C. '' Curulewski Glen Burtnik Studio albums Styx ( 1972 ) Styx II ( 1973 ) The Serpent Is Rising ( 1973 ) Man of Miracles ( 1974
) Equinox ( 1975 ) Crystal Ball ( 1976 ) The Grand Illusion ( 1977 ) Pieces of Eight ( 1978 ) Cornerstone ( 1979 ) Paradise Theatre ( 1981 ) Kilroy Was Here (
1983 ) Edge of the Century ( 1990 ) Brave New World ( 1999 ) Cyclorama ( 2003 ) The Mission ( 2017 ) Live albums Caught in the Act ( 1984 ) Return to Paradise (
1997 ) Arch Allies : Live at Riverport ( 2000 ) Styx World : Live 2001 ( 2001 ) At the River 's Edge : Live in St.
</item 5>

<item 6>
Too Much Time on My Hands
2 on the Top Rock Tracks chart , and No. 4 on the RPM Top Singles chart of Canada . Contents ( hide ) 1 Personnel 2 In popular culture 3 References 4 External
links Personnel ( edit ) Tommy Shaw -- lead vocals , lead guitar Dennis DeYoung -- keyboards , backing vocals James Young -- rhythm guitar , backing vocals
Chuck Panozzo -- bass John Panozzo -- drums In popular culture ( edit ) The song enjoyed a small revival when The Tonight Show host Jimmy Fallon started talking
about the music video in April 2016 . This culminated with the episode on April 29 , 2016 when Fallon showed a shot - by - shot reenactment of the video with
him and guest Paul Rudd on the show . In this video Fallon plays the part of Dennis DeYoung and Rudd plays Tommy Shaw , with A.D. Miles as James Young , Seth
Herzog as John Panozzo , and Gerard Bradford as Chuck Panozzo . Tommy Shaw has commented positively and said he was impressed with Fallon 's vocals . References
( edit ) Jump up ^ `` Styx '' .
</item 6>

<item 7>
Too Much Time on My Hands
You can help Wikipedia by expanding it . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Too_Much_Time_on_My_Hands&oldid=836493612 '' Categories :
1981 singles Styx ( band ) songs Songs written by Tommy Shaw A&M Records singles 1980 songs 1980s rock song stubs Hidden categories : Articles with hAudio
microformats All stub articles Talk Contents About Wikipedia Add links This page was last edited on 15 April 2018 , at 03 : 30 . About Wikipedia
</item 7>

<item 8>
Too Much Time on My Hands
References ( edit ) Jump up ^ `` Styx '' . Jump up ^ `` Jimmy Fallon & Paul Rudd recreate Styx Too Much Time On My Hands video - EW.com '' . Entertainment
Weekly 's EW.com . Jump up ^ `` Jimmy Fallon and Paul Rudd re-create Styx 's ' Too Much Time On My Hands ' video '' . UPI. 30 April 2016 . Jump up ^ Jimmy
Fallon & Paul Rudd Recreate Styx Music Video . 29 April 2016 -- via YouTube . Jump up ^ `` Watch Jimmy Fallon 's and Paul Rudd 's Remake of Styx 's ' Too Much
Time on My Hands ' Video '' . Ultimate Classic Rock . 30 April 2016 . Jump up ^ Danielle Henderson ( 30 April 2016 ) . `` Jimmy Fallon and Paul Rudd Gave Us the
Shot - For - Shot Styx Video Remake We Did n't Know We Needed '' . Esquire . Jump up ^ `` Jimmy Fallon Sings `` Too Much Time on My Hands '' - Styx `` . Styx .
</item 8>

<item 9>
Too Much Time on My Hands
Planned '' ( 1981 ) `` Rockin ' the Paradise '' ( 1981 ) `` Snowblind '' ( 1981 ) `` Mr. Roboto '' ( 1983 ) `` Do n't Let It End '' ( 1983 ) `` High Time '' (
1983 ) `` Music Time '' / `` Heavy Metal Poisoning '' ( 1984 ) `` Love Is the Ritual '' ( 1990 ) `` Show Me the Way '' ( 1991 ) `` Love at First Sight '' ( 1991
) `` Paradise '' ( 1997 ) `` On My Way '' ( 1997 ) `` Waiting for Our Time '' ( 2003 ) `` I Am the Walrus '' ( 2003 ) `` Ca n't Stop Rockin ' '' ( 2009 ) ``
Gone Gone Gone '' ( 2017 ) Related articles Discography Book This 1980s rock song - related article is a stub . You can help Wikipedia by expanding it .
</item 9>

<item 10>
Too Much Time on My Hands
) Regeneration : Volume I & II ( 2011 & 2012 ) Cover albums Big Bang Theory ( 2005 ) Songs `` Best Thing '' ( 1972 ) `` Lady '' ( 1973 ) `` You Need Love '' (
1975 ) `` Lorelei '' ( 1976 ) `` Mademoiselle '' ( 1976 ) `` Crystal Ball '' ( 1977 ) `` Come Sail Away '' ( 1977 ) `` Fooling Yourself ( The Angry Young Man )
'' ( 1978 ) `` Blue Collar Man ( Long Nights ) '' ( 1978 ) `` Sing for the Day '' ( 1979 ) `` Renegade ( 1979 ) `` Babe '' ( 1979 ) `` Why Me '' ( 1980 ) ``
Boat on the River ( 1980 ) `` Borrowed Time '' ( 1980 ) `` Lights '' ( 1980 ) `` First Time '' ( 1980 ) `` The Best of Times '' ( 1981 ) `` Too Much Time on My
Hands '' ( 1981 ) `` Nothing Ever Goes as Planned '' ( 1981 ) `` Rockin ' the Paradise '' ( 1981 ) ``
</item 10>
=== END ITEM id=6a9ceab1f7ac14ab ===

=== ITEM id=20d68afc78e6fe46 ===
Gold sentences and facts:
Sentence 0: The Ring of Fire is a direct result of plate tectonics : the movement and collisions of lithospheric plates . The eastern section of the ring is the result of the Nazca Plate and the Cocos Plate being subducted beneath the westward - moving South American Plate . The Cocos Plate is being subducted beneath the Caribbean Plate , in Central America . A portion of the Pacific Plate and the small Juan de Fuca Plate are being subducted beneath the North American Plate . Along the northern portion , the northwestward - moving Pacific plate is being subducted beneath the Aleutian Islands arc . Farther west , the Pacific plate is being subducted along the Kamchatka Peninsula arcs on south past Japan . The southern portion is more complex , with a number of smaller tectonic plates in collision with the Pacific plate from the Mariana Islands , the Philippines , Bougainville , Tonga , and New Zealand ; this portion excludes Australia , since it lies in the center of its tectonic plate . Indonesia lies between the Ring of Fire along the northeastern islands adjacent to and including New Guinea and the Alpide belt along the south and west from Sumatra , Java , Bali , Flores , and Timor . The famous and very active San Andreas Fault zone of California is a transform fault which offsets a portion of the East Pacific Rise under southwestern United States and Mexico . The motion of the fault generates numerous small earthquakes , at multiple times a day , most of which are too small to be felt . The active Queen Charlotte Fault on the west coast of the Haida Gwaii , British Columbia , has generated three large earthquakes during the 20th century : a magnitude 7 event in 1929 ; a magnitude 8.1 in 1949 ( Canada 's largest recorded earthquake ) ; and a magnitude 7.4 in 1970 .
  Fact 0.0: The Ring of Fire is a direct result of plate tectonics.
  Fact 0.1: Plate tectonics involves the movement and collisions of lithospheric plates.
  Fact 0.2: The eastern section of the Ring of Fire is the result of the Nazca Plate being subducted beneath the westward-moving South American Plate.
  Fact 0.3: The eastern section of the Ring of Fire is the result of the Cocos Plate being subducted beneath the westward-moving South American Plate.
  Fact 0.4: The Cocos Plate is being subducted beneath the Caribbean Plate in Central America.
  Fact 0.5: A portion of the Pacific Plate is being subducted beneath the North American Plate.
  Fact 0.6: The small Juan de Fuca Plate is being subducted beneath the North American Plate.
  Fact 0.7: Along the northern portion, the northwestward-moving Pacific plate is being subducted beneath the Aleutian Islands arc.
  Fact 0.8: Farther west, the Pacific plate is being subducted along the Kamchatka Peninsula arcs on south past Japan.
  Fact 0.9: The southern portion of the Ring of Fire is more complex.
  Fact 0.10: In the southern portion, a number of smaller tectonic plates are in collision with the Pacific plate from the Mariana Islands, the Philippines, Bougainville, Tonga, and New Zealand.
  Fact 0.11: The southern portion of the Ring of Fire excludes Australia.
  Fact 0.12: Australia is excluded from the southern portion since it lies in the center of its tectonic plate.
  Fact 0.13: Indonesia lies between the Ring of Fire along the northeastern islands adjacent to and including New Guinea and the Alpide belt.
  Fact 0.14: The Alpide belt runs along the south and west of Indonesia from Sumatra, Java, Bali, Flores, and Timor.
  Fact 0.15: The San Andreas Fault zone of California is famous and very active.
  Fact 0.16: The San Andreas Fault zone of California is a transform fault.
  Fact 0.17: The San Andreas Fault offsets a portion of the East Pacific Rise under southwestern United States and Mexico.
  Fact 0.18: The motion of the San Andreas Fault generates numerous small earthquakes, at multiple times a day.
  Fact 0.19: Most of the small earthquakes generated by the San Andreas Fault are too small to be felt.
  Fact 0.20: The Queen Charlotte Fault is active.
  Fact 0.21: The Queen Charlotte Fault is on the west coast of the Haida Gwaii, British Columbia.
  Fact 0.22: The Queen Charlotte Fault has generated three large earthquakes during the 20th century.
  Fact 0.23: One of the large Queen Charlotte Fault earthquakes was a magnitude 7 event in 1929.
  Fact 0.24: One of the large Queen Charlotte Fault earthquakes was a magnitude 8.1 event in 1949, which is Canada's largest recorded earthquake.
  Fact 0.25: One of the large Queen Charlotte Fault earthquakes was a magnitude 7.4 event in 1970.

Retrieved items:
<item 1>
Ring of Fire
Sea Scotia Sea Somov Sea Weddell Sea Landlocked seas Aral Sea Caspian Sea Dead Sea Salton Sea Book Related subjects and articles Countries of the Pacific Ring
of Fire Belize Bolivia Brazil Canada Colombia Chile Costa Rica Ecuador East Timor El Salvador Micronesia Fiji Guatemala Honduras Indonesia Japan Kiribati Mexico
New Zealand Nicaragua Palau Papua New Guinea Panama Peru Philippines Russia Samoa Solomon Islands Tonga Tuvalu United States Oceanic Trenches of the Pacific
Ring of Fire Aleutian Trench Bougainville Trench Kermadec Trench Izu Bonin Trench Japan Trench Kurile Trench Mariana Trench Middle America Trench Peru -- Chile
Trench Philippine Trench Ryukyu Trench Tonga Trench Yap Trench Tectonic Plates of the Pacific Ring of Fire Antarctic Plate Australian Plate Caribbean Plate
Cocos Plate Eurasian Plate Explorer Plate Gorda Plate Juan de Fuca Plate Nazca Plate North American Plate Pacific Plate Philippine Sea Plate Rivera Plate South
American Plate Volcanoes of the Pacific Ring of Fire Baker Bulusan Cold Bay Concepción Fuego Fuji Galeras Hood Mount Erebus Krakatoa Mayon Cascade Volcanoes
Merapi Momotombo Novarupta Parícutin Pico de
</item 1>

<item 2>
Ring of Fire
Land areas ( edit ) Papua New Guinea and tectonic plates : Pacific Plate , Australian Plate , Caroline Plate , Banda Sea Plate ( as `` Mer de Banda '' ) ,
Woodlark Plate , Bird 's Head Plate , Maoke Plate , Solomon Sea Plate , North Bismarck Plate , South Bismarck Plate and Manus Plate ( in French ) Puysegur
Trench and Macquarie Ridge Taupo Volcanic Zone Kermadec Islands Tonga Islands Fiji Islands Bougainville and Solomon Islands New Hebrides Arc Bismarck Volcanic
Arc , junction Tanimbar and Kai Islands Lesser Sunda Islands Sunda Arc Andaman and Nicobar Islands Izu - Bonin - Mariana Arc Mariana Islands Bonin Islands Izu
Islands Philippine Mobile Belt Taiwan Ryukyu Islands Japan , Boso Triple Junction Kuril Islands Kamchatka Peninsula American Cordillera North American
Cordillera Aleutian Arc Aleutian Islands Aleutian Range Eastern Alaska , U.S. Northern Cordilleran Volcanic Province Cascade Volcanic Arc and Rio Grande rift
Sierra Nevada ( U.S. )
</item 2>

<item 3>
Ring of Fire
Trans - Mexican Volcanic Belt Central America Volcanic Arc Andes North Volcanic Zone Central Volcanic Zone South Volcanic Zone Austral Volcanic Zone Scotia Arc
South Sandwich Islands Antarctica Antarctic Peninsula Victoria Land See also ( edit ) Earthquakes portal Volcanoes portal 2004 Indian Ocean earthquake and
tsunami 2010 Chile earthquake 2011 Tōhoku earthquake and tsunami Andesite line Geology of the Pacific Northwest Katsuhiko Ishibashi Pacific Rim References (
edit ) Jump up ^ `` Ring of Fire -- Pacific Ring of Fire '' . Geography.about.com. 2010 - 06 - 14 . Retrieved 2010 - 11 - 01 . Jump up ^ `` Ring of Fire '' .
USGS. 2012 - 07 - 24 . Retrieved 2013 - 06 - 13 . Jump up ^ `` Where do earthquakes occur ? '' . USGS. 2013 - 05 - 13 . Archived from the original on 2014 - 08
- 05 . Retrieved 2013 - 06 - 13 .
</item 3>

<item 4>
Ring of Fire
The next most seismically active region ( 5 -- 6 % of earthquakes and 17 % of the world 's largest earthquakes ) is the Alpide belt , which extends from Java to
the northern Atlantic Ocean via the Himalayas and southern Europe . All but three of the world 's 25 largest volcanic eruptions of the last 11,700 years
occurred at volcanoes in the Ring of Fire . The Ring of Fire is a direct result of plate tectonics : the movement and collisions of lithospheric plates . The
eastern section of the ring is the result of the Nazca Plate and the Cocos Plate being subducted beneath the westward - moving South American Plate . The Cocos
Plate is being subducted beneath the Caribbean Plate , in Central America . A portion of the Pacific Plate and the small Juan de Fuca Plate are being subducted
beneath the North American Plate . Along the northern portion , the northwestward - moving Pacific plate is being subducted beneath the Aleutian Islands arc .
Farther west , the Pacific plate is being subducted along the Kamchatka Peninsula arcs on south past Japan .
</item 4>

<item 5>
Ring of Fire
Ring of Fire - wikipedia Ring of Fire Jump to : navigation , search For other uses , see Ring of Fire ( disambiguation ) . The Pacific Ring of Fire Tectonic
plates of the world Great ( magnitude 8.0 + ) earthquakes 1900 -- 2017 , with the majority positioned along the Ring of Fire The Ring of Fire is a major area in
the basin of the Pacific Ocean where a large number of earthquakes and volcanic eruptions occur . In a 40,000 km ( 25,000 mi ) horseshoe shape , it is
associated with a nearly continuous series of oceanic trenches , volcanic arcs , and volcanic belts and plate movements . It has 452 volcanoes ( more than 75 %
of the world 's active and dormant volcanoes ) . The Ring of Fire is sometimes called the circum - Pacific belt . About 90 % of the world 's earthquakes and 81
% of the world 's largest earthquakes occur along the Ring of Fire .
</item 5>

<item 6>
Ring of Fire
The southern portion is more complex , with a number of smaller tectonic plates in collision with the Pacific plate from the Mariana Islands , the Philippines ,
Bougainville , Tonga , and New Zealand ; this portion excludes Australia , since it lies in the center of its tectonic plate . Indonesia lies between the Ring
of Fire along the northeastern islands adjacent to and including New Guinea and the Alpide belt along the south and west from Sumatra , Java , Bali , Flores ,
and Timor . The famous and very active San Andreas Fault zone of California is a transform fault which offsets a portion of the East Pacific Rise under
southwestern United States and Mexico . The motion of the fault generates numerous small earthquakes , at multiple times a day , most of which are too small to
be felt . The active Queen Charlotte Fault on the west coast of the Haida Gwaii , British Columbia , has generated three large earthquakes during the 20th
century : a magnitude 7 event in 1929 ; a magnitude 8.1 in 1949 ( Canada 's largest recorded earthquake ) ; and a magnitude 7.4 in 1970 .
</item 6>

<item 7>
Ring of Fire
Map of the Ring of Fire Ring of Fire , tectonic activity The Ring of Fire at work Physical World Map 2004 - 04 - 01 CIA World Factbook ; Robinson Projection ;
standard parallels 38 ° N and 38 ° S Regions of the world Regions of Africa Central Africa Guinea region Gulf of Guinea Cape Lopez Mayombe Igboland Mbaise
Maputaland Pool Malebo Congo Basin Chad Basin Congolese rainforests Ouaddaï highlands Ennedi Plateau East Africa African Great Lakes Albertine Rift East African
Rift Great Rift Valley Gregory Rift Rift Valley lakes Swahili coast Virunga Mountains Zanj Horn of Africa Afar Triangle Al - Habash Barbara Danakil Alps Danakil
Desert Ethiopian Highlands Gulf of Aden Gulf of Tadjoura Indian Ocean islands Comoros Islands North Africa Maghreb Barbary Coast Bashmur Ancient Libya Atlas
Mountains Nile Valley Cataracts of the Nile Darfur Gulf of Aqaba Lower Egypt Lower Nubia Middle Egypt Nile Delta Nuba Mountains Nubia The Sudans Upper Egypt
Western Sahara West Africa Pepper Coast Gold Coast Slave Coast Ivory Coast Cape Palmas Cape Mesurado Guinea region Gulf of Guinea Niger Basin Guinean Forests of
West
</item 7>

<item 8>
Ring of Fire
Islands Nauru Northern Mariana Islands Wake Island Polynesia Easter Island Hawaiian Islands Cook Islands French Polynesia Austral Islands Gambier Islands
Marquesas Islands Society Islands Tuamotu Kermadec Islands Mangareva Islands Samoa Tokelau Tonga Tuvalu Ring of Fire Regions of South America East Amazon basin
Atlantic Forest Caatinga Cerrado North Caribbean South America West Indies Los Llanos The Guianas Amazon basin Amazon rainforest Gulf of Paria Paria Peninsula
Paraguaná Peninsula Orinoco Delta South Tierra del Fuego Patagonia Pampas Pantanal Gran Chaco Chiquitano dry forests Valdes Peninsula West Andes Tropical Andes
Wet Andes Dry Andes Pariacaca mountain range Altiplano Atacama Desert Latin Hispanic American Cordillera Ring of Fire LAC Polar regions Antarctic Antarctic
Peninsula East Antarctica West Antarctica Eklund Islands Ecozone Extreme points Islands Arctic Arctic Alaska British Arctic Territories Canadian Arctic
Archipelago Finnmark Greenland Northern Canada Northwest Territories Nunavik Nunavut Russian Arctic Sakha Sápmi Yukon North American Arctic Earth 's oceans and
seas Arctic Ocean Amundsen Gulf Barents Sea Beaufort Sea Chukchi Sea East
</item 8>

<item 9>
Ring of Fire
Retrieved 2006 - 10 - 23 . Jump up ^ New Zealand Department of Conservation . `` Central North Island Volcanoes '' . Retrieved 2006 - 10 - 23 . Jump up ^ ``
Contingency Plan for the Auckland Volcanic Field '' ( PDF ) . Auckland Regional Council . Archived from the original ( PDF ) on 2006 - 01 - 16 . ^ Jump up to :
Skiing the Pacific Ring of Fire and Beyong : Antarctica . Retrieved 2007 - 07 - 31 . Jump up ^ `` Mount Erebus Volcano Observatory '' . New Mexico Tech .
Retrieved 2008 - 12 - 29 . External links ( edit ) Wikimedia Commons has media related to Maps of the Pacific Ring of Fire . Historic Earthquakes & Earthquake
Statistics at the United States Geological Survey DESCRIPTION : `` Ring of Fire '' , Plate Tectonics , Sea - Floor Spreading , Subduction Zones , `` Hot Spots
'' at the USGS Cascades Volcano Observatory , Vancouver , Washington Web site . Map of the Ring of Fire Ring of Fire ,
</item 9>

<item 10>
Ring of Fire
Cascade Volcanoes Merapi Momotombo Novarupta Parícutin Pico de Orizaba Pinatubo Popocatépetl Shasta Rainier Ruapehu Nevado del Ruiz Saint Helens Tambora
Taranaki Tungurahua Usu Hoodoo Edziza Tseax The Volcano Meager Garibaldi Cayley Silverthrone Volcano Mountain Other Notable Geographic Features of the Pacific
Ring of Fire Alaska Aleutian Islands American cordillera Andes Antarctica Bali Borneo Bougainville British Columbia California Cascade Range Challenger Deep
Coast Mountains Fais Island Flores Guam Honshū Insular Mountains Java Kamchatka Kurile Islands Luzon Mariana Islands Melanesia Micronesia Mindanao New Guinea
North Island ( NZL ) Northern Mariana Islands Oregon Pacific Coast Range Pacific Ocean Polynesia Rocky Mountains San Andreas Fault Queen Charlotte Fault Sierra
Madre ( PHL ) Sierra Madre ( USA ) Sierra Madre del Sur Sierra Madre Occidental Sierra Madre Oriental South Island ( NZL ) Sulawesi Timor Washington Yap Yukon
Territory Retrieved from `` https://en.wikipedia.org/w/index.
</item 10>
=== END ITEM id=20d68afc78e6fe46 ===
