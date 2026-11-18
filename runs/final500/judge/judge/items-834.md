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

=== ITEM id=c6c6e944ec09e94d ===
Gold sentences and facts:
Sentence 0: The theory is generally associated with Hirschman . He presented a complete theoretical formulation of the strategy . Underdeveloped countries display common characteristics : low levels of GNI per capita and slow GNI per capita growth , large income inequalities and widespread poverty , low levels of productivity , great dependence on agriculture , a backward industrial structure , a high proportion of consumption and low savings , high rates of population growth and dependency burdens , high unemployment and underemployment , technological backwardness and dualism ( existence of both traditional and modern sectors ) . In a less - developed country , these characteristics lead to scarce resources or inadequate infrastructure to exploit these resources . With a lack of investors and entrepreneurs , cash flows can not be directed into various sectors that influence balanced economic growth .
  Fact 0.0: The theory is generally associated with Hirschman.
  Fact 0.1: Hirschman presented a complete theoretical formulation of the strategy.
  Fact 0.2: Underdeveloped countries display common characteristics.
  Fact 0.3: Underdeveloped countries have low levels of GNI per capita.
  Fact 0.4: Underdeveloped countries have slow GNI per capita growth.
  Fact 0.5: Underdeveloped countries have large income inequalities.
  Fact 0.6: Underdeveloped countries have widespread poverty.
  Fact 0.7: Underdeveloped countries have low levels of productivity.
  Fact 0.8: Underdeveloped countries have great dependence on agriculture.
  Fact 0.9: Underdeveloped countries have a backward industrial structure.
  Fact 0.10: Underdeveloped countries have a high proportion of consumption and low savings.
  Fact 0.11: Underdeveloped countries have high rates of population growth and dependency burdens.
  Fact 0.12: Underdeveloped countries have high unemployment and underemployment.
  Fact 0.13: Underdeveloped countries have technological backwardness.
  Fact 0.14: Underdeveloped countries have dualism, meaning both traditional and modern sectors exist.
  Fact 0.15: In a less-developed country, these characteristics lead to scarce resources or inadequate infrastructure to exploit these resources.
  Fact 0.16: In a less-developed country, a lack of investors and entrepreneurs means cash flows cannot be directed into various sectors that influence balanced economic growth.

Retrieved items:
<item 1>
Economy of India
179 Jump up ^ Kumar 2005 , p. 519 Jump up ^ Irfan Habib , Dharma Kumar , Tapan Raychaudhuri ( 1987 ) . The Cambridge Economic History of India ( PDF ) . 1 .
Cambridge University Press . p. 165 . CS1 maint : Multiple names : authors list ( link ) Jump up ^ Kumar 2005 , pp. 520 -- 521 Jump up ^ Roy 2006 , pp. 22 -- 24
Jump up ^ Parthasarathi , Prasannan ( 2011 ) , Why Europe Grew Rich and Asia Did Not : Global Economic Divergence , 1600 -- 1850 , Cambridge University Press ,
p. 45 , ISBN 978 - 1 - 139 - 49889 - 0 Jump up ^ Angus Maddison ( 2007 ) . The World Economy Volume 1 : A Millennial Perspective Volume 2 : Historical
Statistics . Academic Foundation . p. 260 . Jump up ^ Maddison , Angus ( 2007 ) , Contours of the World Economy , 1 -- 2030 AD .
</item 1>

<item 2>
Strategy of unbalanced growth
( hide ) Development Economics : Partial Theories of Development Dual - sector model Big push model Critical minimum effort theory Ragnar Nurkse 's balanced
growth theory Strategy of unbalanced growth Social dualism Technological dualism Fei -- Ranis model of economic growth Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Strategy_of_unbalanced_growth&oldid=838124154 '' Categories : Economic growth Hidden categories : Wikipedia articles
that are incomprehensible from August 2016 All Wikipedia articles that are incomprehensible Articles lacking in - text citations from January 2012 All articles
lacking in - text citations All articles with unsourced statements Articles with unsourced statements from February 2013 Wikipedia articles needing
clarification from February 2013 Articles with unsourced statements from March 2012 All articles with specifically marked weasel - worded phrases Articles with
specifically marked weasel - worded phrases from February 2013 Talk Contents About Wikipedia Deutsch Français 中文 Edit links This page was last edited on 25
April 2018 , at 02 : 21 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 2>

<item 3>
Strategy of unbalanced growth
The services provided should be subject to public control . The services can not be imported . The investment needed to provide services should be characterized
by some unevenness as well as by high capital output ratio . Development via capital imbalances ( edit ) The strategy of unbalanced growth has been discussed
within the frameworks of development through shortage of SOC and development through excess of SOC . In the first case , the country invests in direct
productive activities ( DPA ) . Direct productive activity increases demand for SOC , inducing investment . In the second case , SOC expands , which reduces the
cost of services , inducing investment in DPA . The cost of producing any unit of output of DPA is inversely proportional to SOC . The economy 's major
objective is to attain increasing output of DPA . One of the paradoxes of development is that poor countries can not afford to be economical . According to
Hirschman , resources are not scarce per se , but the ability to employ those resources may be lacking . To explain unbalanced growth , Hirschman assumes that
the country invests in either DPA or SOC .
</item 3>

<item 4>
Strategy of unbalanced growth
Paul Streeten , `` Unbalanced Growth . '' Economic Integration , Sythoff , Leiden ( Netherlands ) , 1961 . Reprinted in A.N. Agarwala and S.P. Singh , ( eds . )
, Accelerating Investment in Developing Economies ( London : Oxford University Press , 1969 ) .
</item 4>

<item 5>
Strategy of unbalanced growth
Strategy of unbalanced growth - Wikipedia Strategy of unbalanced growth Jump to : navigation , search This article may be incomprehensible or very hard to
understand . In particular , it uses both very general statements which are not further clarified or contextualized , and technical jargon that is not explained
. Please help by rewording it if the intended meaning can be determined . The talk page may have details . ( August 2016 ) ( Learn how and when to remove this
template message ) This article includes a list of references , related reading or external links , but its sources remain unclear because it lacks inline
citations . Please help to improve this article by introducing more precise citations . ( January 2012 ) ( Learn how and when to remove this template message )
Unbalanced growth is a natural path of economic development . Situations that countries are in at any one point in time reflect their previous investment
decisions and development . Accordingly , at any point in time desirable investment programs that are not balanced investment packages may still advance welfare
. Unbalanced investment can complement or correct existing imbalances . Once such an investment is made , a new imbalance is likely to appear , requiring
further compensating investments .
</item 5>

<item 6>
Strategy of unbalanced growth
In a less - developed country , these characteristics lead to scarce resources or inadequate infrastructure to exploit these resources . With a lack of
investors and entrepreneurs , cash flows can not be directed into various sectors that influence balanced economic growth . Hirschman contends that deliberate
unbalancing of the economy according to the strategy is the best method of development and if the economy is to be kept moving ahead , the task of development
policy is to maintain tension , disproportions and disequilibrium . Balanced growth should not be the goal but rather the maintenance of existing imbalances ,
which can be seen from profit and losses . Therefore , the sequence that leads away from equilibrium is precisely an ideal pattern for development . Unequal
development of various sectors often generates conditions for rapid development . More - developed industries provide undeveloped industries an incentive to
grow . Hence , development of underdeveloped countries should be based on this strategy . The path of unbalanced growth is described by three phases :
Complementary Induced investment External economies Singer believed that desirable investment programs always exist within a country that represent unbalanced
investment to complement the existing imbalance . These investments create a new imbalance , requiring another balancing investment .
</item 6>

<item 7>
Strategy of unbalanced growth
Therefore , growth need not take place in a balanced way . Supporters of the unbalanced growth doctrine include Albert O. Hirschman , Hans Singer , Paul
Streeten , Marcus Fleming , Prof. Rostov and J. Sheehan . Contents ( hide ) 1 Introduction 2 Complementarity 3 Induced investment 4 External economies 5 Social
Overhead Capital 6 Development via capital imbalances 7 Backward and forward linkages 8 Linkages and last industries 8.1 Disadvantages 9 Further reading 10
Notes 11 References Introduction ( edit ) The theory is generally associated with Hirschman . He presented a complete theoretical formulation of the strategy .
Underdeveloped countries display common characteristics : low levels of GNI per capita and slow GNI per capita growth , large income inequalities and widespread
poverty , low levels of productivity , great dependence on agriculture , a backward industrial structure , a high proportion of consumption and low savings ,
high rates of population growth and dependency burdens , high unemployment and underemployment , technological backwardness and dualism ( existence of both
traditional and modern sectors ) .
</item 7>

<item 8>
Strategy of unbalanced growth
These investments create a new imbalance , requiring another balancing investment . One sector will always grow faster than another , so the need for unbalanced
growth will continue as investments must complement existing imbalance . Hirschman states `` If the economy is to be kept moving ahead , the task of development
policy is to maintain tensions , disproportions and disequilibrium '' . This situation exists for all societies , developed or underdeveloped . Complementarity
( edit ) Complementarity is a situation where increased production of one good or service builds up demand for the second good or service . When the second
product is privately produced , this demand will lead to imports or higher domestic production of the second product , as it will be in the interests of the
producers to do so . Otherwise , the increased demand takes the form of political pressure . This is the case for such public services such as law and order ,
education , water and electricity that can not reasonably be imported . Induced investment ( edit ) Complementarity allows investment in one industry or sector
to encourage investment in others . This concept of induced investment is like a multiplier , because each investment triggers a series of subsequent events .
</item 8>

<item 9>
Strategy of unbalanced growth
11 , No. 2 ( Jun. , 1959 ) , pp. 167 -- 190 Notes ( edit ) References ( edit ) Hirschman , Albert O . `` The Strategy of Economic Development , '' in Agarwal ,
A.N. and Singh , S.P. ( eds ) , Accelerating Investment in Developing Economies ( London Oxford Press , 1969 ) Thirwall , A.P. Growth and Development : With
special reference to developing economies ( London : Macmillian Press ltd ... , sixth edition . 1999 ) , pp237 -- 242 . Scitovsky , Tibor . Papers on welfare
and Growth ( London George Allen and Unwin ltd ... 1965 ) Refer to 5th paper entitled `` Growth -- Balanced and Unbalanced '' . Nath , S.K. , `` Balanced Growth
. '' in Livingstone ( ed ) , Economic Policy for Development ( Harmodworth , Middlesex : Penguin Books ltd. 1971 ) Han Singer , `` The Concept of Balanced
Growth and Economic Development ; Theory and Facts , '' University of Texas Conference on Economic Development , April 1958 .
</item 9>

<item 10>
Graph query row:
entity: Albert O. Hirschman
relationship: RELATED_TO
related: Strategy of unbalanced growth
</item 10>
=== END ITEM id=c6c6e944ec09e94d ===

=== ITEM id=6bb6e2d3471b8b94 ===
Gold sentences and facts:
Sentence 0: The Compromise of 1877 was an informal , unwritten deal that settled the intensely disputed 1876 U.S. presidential election . It resulted in the United States federal government pulling the last troops out of the South , and formally ended the Reconstruction Era . Through the Compromise , Republican Rutherford B. Hayes was awarded the White House over Democrat Samuel J. Tilden on the understanding that Hayes would remove the federal troops whose support was essential for the survival of Republican state governments in South Carolina , Florida and Louisiana . The compromise involved Democrats who controlled the House of Representatives allowing the decision of the Electoral Commission to take effect . The outgoing president , Republican Ulysses S. Grant , removed the soldiers from Florida . As president , Hayes removed the remaining troops from South Carolina and Louisiana . As soon as the troops left , many white Republicans also left , and the `` Redeemer '' Democrats took control . They already dominated other state governments in the South . What was exactly agreed is somewhat contested as the documentation is insufficient .
  Fact 0.0: The Compromise of 1877 was an informal, unwritten deal.
  Fact 0.1: The Compromise of 1877 settled the intensely disputed 1876 U.S. presidential election.
  Fact 0.2: The Compromise resulted in the United States federal government pulling the last troops out of the South.
  Fact 0.3: The Compromise formally ended the Reconstruction Era.
  Fact 0.4: Through the Compromise, Republican Rutherford B. Hayes was awarded the White House over Democrat Samuel J. Tilden.
  Fact 0.5: The White House was awarded to Hayes on the understanding that Hayes would remove the federal troops.
  Fact 0.6: The federal troops' support was essential for the survival of Republican state governments in South Carolina, Florida and Louisiana.
  Fact 0.7: The compromise involved Democrats who controlled the House of Representatives.
  Fact 0.8: The compromise involved the Democrats allowing the decision of the Electoral Commission to take effect.
  Fact 0.9: The outgoing president, Republican Ulysses S. Grant, removed the soldiers from Florida.
  Fact 0.10: As president, Rutherford B. Hayes removed the remaining troops from South Carolina.
  Fact 0.11: As president, Rutherford B. Hayes removed the remaining troops from Louisiana.
  Fact 0.12: When the troops left, many white Republicans also left.
  Fact 0.13: When the troops left, the 'Redeemer' Democrats took control.
  Fact 0.14: The 'Redeemer' Democrats already dominated other state governments in the South.
  Fact 0.15: What was exactly agreed in the Compromise of 1877 is somewhat contested.
  Fact 0.16: The documentation of the Compromise of 1877 is insufficient.

Retrieved items:
<item 1>
Compromise of 1877
( Oxford University Press , 1982 ) pp. 417 -- 50 . Downs , Gregory P. ( 2012 ) . `` The Mexicanization of American Politics : The United States ' Transnational
Path from Civil War to Stabilization '' . The American Historical Review . 117 ( 2 ) : 387 -- 409 . doi : 10.1086 / ahr. 117.2. 387 . Frantz , Edward O . The
Door of Hope : Republican Presidents and the First Southern Strategy , 1877 -- 1933 ( University Press of Florida . 2011 ) Peskin , Allan ( 1973 ) . `` Was
There a Compromise of 1877 '' . The Journal of American History . 60 ( 1 ) : 63 -- 75 . doi : 10.2307 / 2936329 . ISSN 1936 - 0967 . JSTOR 2936329 -- via JSTOR
. ( Registration required ( help ) ) . Polakoff , Keith Ian .
</item 1>

<item 2>
Compromise of 1877
The outgoing president , Republican Ulysses S. Grant , removed the soldiers from Florida . As president , Hayes removed the remaining troops from South Carolina
and Louisiana . As soon as the troops left , many white Republicans also left , and the `` Redeemer '' Democrats took control . They already dominated other
state governments in the South . What was exactly agreed is somewhat contested as the documentation is insufficient . Black Republicans felt betrayed as they
lost power and were subject to discrimination and harassment to suppress their voting . by 1905 most black men were effectively disenfranchised by state
legislatures in every southern state . Contents ( hide ) 1 Terms of compromise 2 Results 3 Interpretations 4 References 5 Further reading 6 External links Terms
of compromise ( edit ) The compromise essentially stated that Southern Democrats would acknowledge Hayes as president , but only on the understanding that
Republicans would meet certain demands . The following elements are generally said to be the points of the compromise : The removal of all U.S. military forces
from the former Confederate states . At the time , U.S.
</item 2>

<item 3>
Compromise of 1877
He tightened military security , and nobody marched on Washington . Hayes was peacefully inaugurated . Points 1 and 2 of the compromise took effect . Hayes had
already announced his support for the restoration of `` home rule , '' which would involve federal troop removal , before the election . It was not unusual ,
nor unexpected , for a president , especially one so narrowly elected , to select a cabinet member favored by the other party . Points 3 and 4 were never
enacted ; it is possible there was no firm agreement about them . Whether by informal deal or simply reassurances already in line with Hayes 's announced plans
, talks with Southern Democrats satisfied the worries of many . This prevented a Congressional filibuster that had threatened to extend resolution of the
election dispute beyond Inauguration Day 1877 . Interpretations ( edit ) C. Vann Woodward ( 1951 ) argued for an economic interpretation , saying that emerging
business and industry interests of the New South found common ground with Republican businessmen , particularly with the railroads .
</item 3>

<item 4>
Compromise of 1877
Hayes Presidential Library Southernspaces.org : `` Corporations , Corruption , and the Modern Lobby : A Gilded Age Story of the West and the South in Washington
, D.C. '' -- by Richard White in Southern Spaces ( 16 April 2009 ) . Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Compromise_of_1877&oldid=835984417 '' Categories : Reconstruction Era African - American history History of the
Southern United States 1877 in the United States Rutherford B. Hayes Hidden categories : All articles with unsourced statements Articles with unsourced
statements from November 2016 Articles with unsourced statements from August 2017 Pages with login required references or sources Talk Contents About Wikipedia
Español Français 한국어 Lietuvių 日本 語 Português Русский Simple English Svenska Edit links This page was last edited on 11 April 2018 , at 23 : 51 . Text is
available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and
Privacy Policy .
</item 4>

<item 5>
Compromise of 1877
military forces from the former Confederate states . At the time , U.S. troops remained in only Louisiana , South Carolina , and Florida , but the Compromise
completed their withdrawal from the region . The appointment of at least one Southern Democrat to Hayes ' cabinet . ( David M. Key of Tennessee was appointed as
Postmaster General . ) The construction of another transcontinental railroad using the Texas and Pacific in the South ( this had been part of the `` Scott Plan
, '' proposed by Thomas A. Scott of the Pennsylvania Railroad ; he had initiated negotiations resulting in the final compromise ) . Legislation to help
industrialize the South and restore its economy following Reconstruction and the Civil War . The right to deal with blacks without northern interference . In
exchange , Democrats would accept the Republican Hayes as president by not employing the filibuster during the joint session of Congress needed to confirm the
election . Results ( edit ) After the Compromise , a few Democrats complained loudly that Tilden had been cheated . There was talk of forming armed units that
would march on Washington , but President Grant was ready for that . He tightened military security , and nobody marched on Washington . Hayes was peacefully
inaugurated .
</item 5>

<item 6>
Compromise of 1877
, Region , Race , and Reconstruction : Essays in Honor of C. Van Woodward ( New York , 1982 ) , 417 - 50 Jump up ^ Michael Les Benedict , `` Southern Democrats
in the Crisis of 1876 - 1877 : A Reconsideration of Reunion and Reaction . '' Journal of Southern History ( 1980 ) : 489 - 524 . Further reading ( edit )
Benedict , Michael Les . `` Southern Democrats in the Crisis of 1876 - 1877 : A Reconsideration of Reunion and Reaction . '' Journal of Southern History ( 1980
) : 489 - 524 . in JSTOR Clendenen , Clarence C. ( October 1969 ) . `` President Hayes ' `` Withdrawal '' of the Troops : An Enduring Myth `` . The South
Carolina Historical Magazine . 70 ( 4 ) : 240 -- 250 . DeSantis , Vincent P. `` Rutherford B. Hayes and the Removal of the Troops and the End of Reconstruction
'' in Region , Race and Reconstruction edited by Morgan Kousser and James McPherson .
</item 6>

<item 7>
Compromise of 1877
U of Michigan Press . p. 211 . Jump up ^ C. Vann Woodward ( 1991 ) . Reunion and Reaction : The Compromise of 1877 and the End of Reconstruction . Oxford UP .
pp. 200 -- 2 ` . Jump up ^ Downs , 2012 Jump up ^ C. Vann Woodward ( 1991 ) . Reunion and Reaction : The Compromise of 1877 and the End of Reconstruction .
Oxford UP . pp. 200 -- 2 ` . ^ Jump up to : Allan Peskin , `` Was There a Compromise of 1877 ? '' , The Journal of American History Vol. 60 , No. 1 ( Jun. ,
1973 ) , pp. 63 - 75 , via JSTOR Jump up ^ Vincent P. DeSantis , `` Rutherford B. Hayes and the Removal of the Troops and the End of Reconstruction , '' in J.
Morgan Kousser and James M. McPherson , eds.
</item 7>

<item 8>
Compromise of 1877
Downs explores how Mexicanization was roundly rejected and stability was achieved . Whatever deals may or may not have taken place on the side , in formal legal
terms , the election of 1876 was not decided by such acts , but by the official vote of Congress to accept the recommendations of the Electoral Commission they
themselves had set up as a way out of the election impasse . The expectation in setting up the committee had been that its decisions would be accepted by
Congress . It was only when certain Democrats disagreed with the commission 's decisions in favor of Hayes that this arrangement was jeopardized . This
Democratic group threatened a filibuster ( opposed by Republicans and Congressional Democratic leadership as well ) that would prevent the agreed - upon vote
from taking place . Discussions of the points in the alleged compromise were related to persuading key Democrats against accepting a filibuster . The very
threat of a filibuster -- a measure used by a minority to prevent a vote -- indicates that there were already sufficient votes for accepting the commission 's
recommendations .
</item 8>

<item 9>
Compromise of 1877
Compromise of 1877 - wikipedia Compromise of 1877 Jump to : navigation , search A political cartoon by Joseph Keppler depicts Roscoe Conkling as a character
Mephistopheles ( the Devil ) while Rutherford B. Hayes strolls off with the prize of the `` Solid South '' depicted as a woman . The caption quotes Goethe : ``
Unto that Power he doth belong Which only doeth Right while ever willing Wrong . '' The Compromise of 1877 was an informal , unwritten deal that settled the
intensely disputed 1876 U.S. presidential election . It resulted in the United States federal government pulling the last troops out of the South , and formally
ended the Reconstruction Era . Through the Compromise , Republican Rutherford B. Hayes was awarded the White House over Democrat Samuel J. Tilden on the
understanding that Hayes would remove the federal troops whose support was essential for the survival of Republican state governments in South Carolina ,
Florida and Louisiana . The compromise involved Democrats who controlled the House of Representatives allowing the decision of the Electoral Commission to take
effect . The outgoing president , Republican Ulysses S. Grant , removed the soldiers from Florida .
</item 9>

<item 10>
Compromise of 1877
References ( edit ) Jump up ^ Michael Les Benedict , `` Southern Democrats in the Crisis of 1876 - 1877 : A Reconsideration of Reunion and Reaction . '' Journal
of Southern History ( 1980 ) : 489 - 524 . in JSTOR Jump up ^ Jones , Stephen A. ; Freedman , Eric ( 2011 ) . Presidents and Black America . CQ Press . p. 218 .
ISBN 9781608710089 . In an eleventh - hour compromise between party leaders - considered the `` Great Betrayal '' by many blacks and southern Republicans ...
Jump up ^ Woodward , C. Vann ( 1966 ) . Reunion and Reaction : The Compromise of 1877 and the End of Reconstruction . Boston : Little , Brown and Company . pp.
169 -- 171 . Jump up ^ Donald Richard Deskins ; Hanes Walton ; Sherman C. Puckett ( 2010 ) . Presidential Elections , 1789 - 2008 : County , State , and
National Mapping of Election Data . U of Michigan Press . p. 211 .
</item 10>
=== END ITEM id=6bb6e2d3471b8b94 ===

=== ITEM id=81bb4384859bc55a ===
Gold sentences and facts:
Sentence 0: Country Name of animal Scientific name Pictures Ref . Algeria Fennec fox Vulpes zerda Angola Red - crested turaco ( national bird ) Tauraco erythrolophus Anguilla Zenaida dove Zenaida aurita Antigua and Barbuda Fallow deer ( national animal ) Dama dama Frigate ( national bird ) Fregata magnificens Hawksbill turtle ( national sea creature ) Eretmochelys imbricata Argentina Rufous hornero Furnarius rufus Australia Red kangaroo ( national animal ) Macropus rufus Emu ( national bird ) Dromaius novaehollandiae Austria Black eagle Ictinaetus malaiensis Azerbaijan Karabakh horse Equus ferus caballus Bangladesh Royal Bengal tiger ( national animal ) Panthera tigris tigris Magpie robin ( national bird ) Copsychus saularis Ilish ( national fish ) Tenualosa ilisha Belarus European bison Bison bonasus Belgium Lion ( heraldic Leo Belgicus ) Panthera leo Belize Baird 's tapir ( national animal ) Tapirus bairdii Keel - billed toucan ( national bird ) Ramphastos sulfuratus Bhutan Druk Mythical Takin Budorcas taxicolor Brazil Rufous - bellied thrush Turdus rufiventris Cambodia Kouprey Bos sauveli Canada North American beaver ( sovereignty animal symbol ) Castor canadensis Canadian horse ( national horse ) Equus ferus caballus China Giant panda ( national animal ) Ailuropoda melanoleuca Chinese dragon ( national animal ) Mythical Red - crowned crane ( national bird ) Grus japonensis Democratic Republic of the Congo Okapi Okapia johnstoni Colombia Andean condor Vultur gryphus Costa Rica Yigüirro ( national bird ) Turdus grayi White - tailed deer ( national animal ) Odocoileus virginianus West Indian manatee ( national aquatic animal ) Trichechus manatus Croatia Pine marten Martes martes Cuba Cuban trogon Priotelus temnurus Cyprus Cypriot mouflon Ovis orientalis Czech Republic Double - tailed lion Mythical Denmark Mute swan ( national bird ) Cygnus olor Small tortoiseshell ( national butterfly ) Aglais urticae Egypt Steppe eagle Aquila nipalensis Estonia Barn swallow ( national bird ) Hirundo rustica Eritrea Arabian camel Camelus dromedarius Ethiopia Lion Panthera leo Faroe Islands Eurasian oystercatcher ( national bird ) Haematopus ostralgus Finland Brown bear ( national animal ) Ursus arctos Whooper swan ( national bird ) Cygnus cygnus Holly blue ( national butterfly ) Celastrina argiolus European perch ( national fish ) Perca fluviatilis Seven - spot ladybird ( national insect ) Coccinella septempunctata France Gallic rooster Gallus gallus domesticus Germany Unspecified Eagle Members of Accipitridae Greece Phoenix Mythical Dolphins Delphinus Guatemala Quetzal Pharomachrus mocinno Haiti Hispaniolan Trogan Priotelus roseigaster Honduras White - tailed deer Odocoileus virginianus Hungary Turul Mythical ( most probably Falco cherrug ) Iceland Gyrfalcon Falco rusticolus India Bengal tiger ( National predator ) Panthera tigris tigris Indian peafowl ( National bird ) Pavo cristatus King cobra ( National reptile ) Ophiophagus hannah Ganges river dolphin ( National aquatic animal ) Platanista gangetica gangetica Indian elephant ( National heritage animal ) Elephas maximus indicus Indonesia Komodo dragon ( national animal ) Varanus komodoensis Garuda ( historic national animal ) Mythical Iran Persian lion Panthera leo persica Iraq Chukar partridge Alectoris chukar Goat ( KRG ) Capra aegagrus Israel Hoopoe ( national bird ) Upupa epops Jamaica Red - billed streamertail Trochilus polytmus Japan Green pheasant ( national bird ) Phasianus versicolor Carp ( national fish ) Cyprinus carpio Latvia White wagtail ( national bird ) Motacilla alba Two - spotted ladybird ( national insect ) Adalia bipunctata Lithuania White stork Ciconia ciconia Macedonia Lion ( in Macedonian heraldry ) Panthera leo Madagascar Zebu ( in Malagasy heraldry ) Bos primigenius indicus Ring - tailed lemur ( national animal ) Lemur catta Malaysia Malayan tiger ( national animal ) Panthera tigris jacksoni Malta Pharaoh Hound ( national animal ) Canis lupus familiaris ( 1 ) Moldova Aurochs ( national animal ) Bos primigenius Mauritius Dodo Raphus cucullatus Mexico Golden eagle ( national animal ) Aquila chrysaetos Xoloitzcuintli ( national dog ) Canis lupus familiaris Jaguar ( national mammal ) Panthera onca Grasshopper ( national arthropod ) Sphenarium purpurascens Vaquita ( national marine mammal ) Phocoena sinus Nepal Cow ( national animal ) Bos taurus indicus Danphe ( national bird ) Lophophorus impejanus Netherlands Lion ( heraldic symbol ) Panthera leo Black - tailed godwit ( national bird ) Limosa limosa New Zealand Kiwi Apteryx sp . North Korea Chollima Mythical Norway Lion ( royal national animal ) Panthera leo White - throated dipper ( national bird ) Cinclus cinclus Fjord horse ( national horse ) Equus ferus caballus Oman Arabian oryx ( national animal ) Oryx leucoryx Pakistan Markhor ( National animal ) Capra falconeri Indus river dolphin ( National aquatic mammal ) Platanista indicus minor minor Chukar ( National bird ) Alectoris chukar Indus crocodile ( National reptile ) Crocodylus palustris Shaheen falcon ( Heritage bird ) Falco peregrinus peregrinator Snow leopard ( National predator ) Panthera uncia Panama Harpy eagle Harpia harpyja Papua New Guinea Dugong ( national marine mammal ) Dugong dugon Peru Vicuña ( national animal ) Vicugna vicugna Philippines Carabao ( national animal ) Bubalus bubalis Philippine eagle ( national bird ) Pithecophaga jefferyi Poland White - tailed eagle ( national bird ) Haliaeetus albicilla White stork ( national bird ) Ciconia ciconia European bison ( national animal ) Bison bonasus Qatar Arabian oryx Oryx leucoryx Romania Lynx Lynx lynx Russia Eurasian brown bear Ursus arctos arctos Rwanda Leopard Panthera pardus Saint Vincent and the Grenadines Saint Vincent amazon Amazona guildingii Serbia Wolf Canis lupus lupus White eagle Mythical Lynx Lynx lynx Somalia Leopard ( national animal ) Panthera pardus South Africa Springbok ( national animal ) Antidorcas marsupialis Blue crane ( national bird ) Anthropoides paradiseus Galjoen ( national fish ) Dichistius capensis South Korea Siberian tiger Panthera tigris altaica Korean magpie ( national bird ) Pica ( pica ) sericea Sri Lanka Sri Lankan junglefowl ( national bird ) Gallus lafayettii Tanzania Giraffe Giraffa camelopardalis Thailand Elephant Elephas maximus United Arab Emirates Arabian oryx ( national animal ) Oryx leucoryx Peregrine falcon ( national bird ) Falco peregrinus United Kingdom Lion ( United Kingdom , England , Scotland , Wales ) Panthera leo Bulldog ( Britain as `` British Bulldog '' ) Canis lupus familiaris Red kite ( Wales ) Milvus milvus Unicorn ( Scotland ) Mythical Y Ddraig Goch ( Welsh Dragon Wales ) Mythical United States Bald eagle ( national bird ) Haliaeetus leucocephalus American bison ( national mammal ) Bison bison
  Fact 0.0: In Algeria, the Fennec fox (Vulpes zerda) is listed as the animal.
  Fact 0.1: In Angola, the Red-crested turaco (Tauraco erythrolophus) is listed as the national bird.
  Fact 0.2: In Anguilla, the Zenaida dove (Zenaida aurita) is listed.
  Fact 0.3: In Antigua and Barbuda, the Fallow deer (Dama dama) is listed as the national animal.
  Fact 0.4: In Antigua and Barbuda, the Frigate (Fregata magnificens) is listed as the national bird.
  Fact 0.5: In Antigua and Barbuda, the Hawksbill turtle (Eretmochelys imbricata) is listed as the national sea creature.
  Fact 0.6: In Argentina, the Rufous hornero (Furnarius rufus) is listed.
  Fact 0.7: In Australia, the Red kangaroo (Macropus rufus) is listed as the national animal.
  Fact 0.8: In Australia, the Emu (Dromaius novaehollandiae) is listed as the national bird.
  Fact 0.9: In Austria, the Black eagle (Ictinaetus malaiensis) is listed.
  Fact 0.10: In Azerbaijan, the Karabakh horse (Equus ferus caballus) is listed.
  Fact 0.11: In Bangladesh, the Royal Bengal tiger (Panthera tigris tigris) is listed as the national animal.
  Fact 0.12: In Bangladesh, the Magpie robin (Copsychus saularis) is listed as the national bird.
  Fact 0.13: In Bangladesh, the Ilish (Tenualosa ilisha) is listed as the national fish.
  Fact 0.14: In Belarus, the European bison (Bison bonasus) is listed.
  Fact 0.15: In Belgium, the Lion (Panthera leo) is listed as the heraldic Leo Belgicus.
  Fact 0.16: In Belize, Baird's tapir (Tapirus bairdii) is listed as the national animal.
  Fact 0.17: In Belize, the Keel-billed toucan (Ramphastos sulfuratus) is listed as the national bird.
  Fact 0.18: In Bhutan, the Druk is listed as a mythical animal.
  Fact 0.19: In Bhutan, the Takin (Budorcas taxicolor) is listed.
  Fact 0.20: In Brazil, the Rufous-bellied thrush (Turdus rufiventris) is listed.
  Fact 0.21: In Cambodia, the Kouprey (Bos sauveli) is listed.
  Fact 0.22: In Canada, the North American beaver (Castor canadensis) is listed as the sovereignty animal symbol.
  Fact 0.23: In Canada, the Canadian horse (Equus ferus caballus) is listed as the national horse.
  Fact 0.24: In China, the Giant panda (Ailuropoda melanoleuca) is listed as the national animal.
  Fact 0.25: In China, the Chinese dragon is listed as a mythical national animal.
  Fact 0.26: In China, the Red-crowned crane (Grus japonensis) is listed as the national bird.
  Fact 0.27: In the Democratic Republic of the Congo, the Okapi (Okapia johnstoni) is listed.
  Fact 0.28: In Colombia, the Andean condor (Vultur gryphus) is listed.
  Fact 0.29: In Costa Rica, the Yigüirro (Turdus grayi) is listed as the national bird.
  Fact 0.30: In Costa Rica, the White-tailed deer (Odocoileus virginianus) is listed as the national animal.
  Fact 0.31: In Costa Rica, the West Indian manatee (Trichechus manatus) is listed as the national aquatic animal.
  Fact 0.32: In Croatia, the Pine marten (Martes martes) is listed.
  Fact 0.33: In Cuba, the Cuban trogon (Priotelus temnurus) is listed.
  Fact 0.34: In Cyprus, the Cypriot mouflon (Ovis orientalis) is listed.
  Fact 0.35: In the Czech Republic, the Double-tailed lion is listed as a mythical animal.
  Fact 0.36: In Denmark, the Mute swan (Cygnus olor) is listed as the national bird.
  Fact 0.37: In Denmark, the Small tortoiseshell (Aglais urticae) is listed as the national butterfly.
  Fact 0.38: In Egypt, the Steppe eagle (Aquila nipalensis) is listed.
  Fact 0.39: In Estonia, the Barn swallow (Hirundo rustica) is listed as the national bird.
  Fact 0.40: In Eritrea, the Arabian camel (Camelus dromedarius) is listed.
  Fact 0.41: In Ethiopia, the Lion (Panthera leo) is listed.
  Fact 0.42: In the Faroe Islands, the Eurasian oystercatcher (Haematopus ostralgus) is listed as the national bird.
  Fact 0.43: In Finland, the Brown bear (Ursus arctos) is listed as the national animal.
  Fact 0.44: In Finland, the Whooper swan (Cygnus cygnus) is listed as the national bird.
  Fact 0.45: In Finland, the Holly blue (Celastrina argiolus) is listed as the national butterfly.
  Fact 0.46: In Finland, the European perch (Perca fluviatilis) is listed as the national fish.
  Fact 0.47: In Finland, the Seven-spot ladybird (Coccinella septempunctata) is listed as the national insect.
  Fact 0.48: In France, the Gallic rooster (Gallus gallus domesticus) is listed.
  Fact 0.49: In Germany, the Unspecified Eagle (a member of Accipitridae) is listed.
  Fact 0.50: In Greece, the Phoenix is listed as a mythical animal.
  Fact 0.51: In Greece, the Dolphins (Delphinus) are listed.
  Fact 0.52: In Guatemala, the Quetzal (Pharomachrus mocinno) is listed.
  Fact 0.53: In Haiti, the Hispaniolan Trogan (Priotelus roseigaster) is listed.
  Fact 0.54: In Honduras, the White-tailed deer (Odocoileus virginianus) is listed.
  Fact 0.55: In Hungary, the Turul is listed as a mythical animal, most probably Falco cherrug.
  Fact 0.56: In Iceland, the Gyrfalcon (Falco rusticolus) is listed.
  Fact 0.57: In India, the Bengal tiger (Panthera tigris tigris) is listed as the National predator.
  Fact 0.58: In India, the Indian peafowl (Pavo cristatus) is listed as the National bird.
  Fact 0.59: In India, the King cobra (Ophiophagus hannah) is listed as the National reptile.
  Fact 0.60: In India, the Ganges river dolphin (Platanista gangetica gangetica) is listed as the National aquatic animal.
  Fact 0.61: In India, the Indian elephant (Elephas maximus indicus) is listed as the National heritage animal.
  Fact 0.62: In Indonesia, the Komodo dragon (Varanus komodoensis) is listed as the national animal.
  Fact 0.63: In Indonesia, the Garuda is listed as the historic national animal, a mythical one.
  Fact 0.64: In Iran, the Persian lion (Panthera leo persica) is listed.
  Fact 0.65: In Iraq, the Chukar partridge (Alectoris chukar) is listed.
  Fact 0.66: In Iraq, the Goat (Capra aegagrus) is listed for KRG.
  Fact 0.67: In Israel, the Hoopoe (Upupa epops) is listed as the national bird.
  Fact 0.68: In Jamaica, the Red-billed streamertail (Trochilus polytmus) is listed.
  Fact 0.69: In Japan, the Green pheasant (Phasianus versicolor) is listed as the national bird.
  Fact 0.70: In Japan, the Carp (Cyprinus carpio) is listed as the national fish.
  Fact 0.71: In Latvia, the White wagtail (Motacilla alba) is listed as the national bird.
  Fact 0.72: In Latvia, the Two-spotted ladybird (Adalia bipunctata) is listed as the national insect.
  Fact 0.73: In Lithuania, the White stork (Ciconia ciconia) is listed.
  Fact 0.74: In Macedonia, the Lion (Panthera leo) is listed as the animal in Macedonian heraldry.
  Fact 0.75: In Madagascar, the Zebu (Bos primigenius indicus) is listed as the animal in Malagasy heraldry.
  Fact 0.76: In Madagascar, the Ring-tailed lemur (Lemur catta) is listed as the national animal.
  Fact 0.77: In Malaysia, the Malayan tiger (Panthera tigris jacksoni) is listed as the national animal.
  Fact 0.78: In Malta, the Pharaoh Hound (Canis lupus familiaris) is listed as the national animal, with reference 1.
  Fact 0.79: In Moldova, the Aurochs (Bos primigenius) is listed as the national animal.
  Fact 0.80: In Mauritius, the Dodo (Raphus cucullatus) is listed.
  Fact 0.81: In Mexico, the Golden eagle (Aquila chrysaetos) is listed as the national animal.
  Fact 0.82: In Mexico, the Xoloitzcuintli (Canis lupus familiaris) is listed as the national dog.
  Fact 0.83: In Mexico, the Jaguar (Panthera onca) is listed as the national mammal.
  Fact 0.84: In Mexico, the Grasshopper (Sphenarium purpurascens) is listed as the national arthropod.
  Fact 0.85: In Mexico, the Vaquita (Phocoena sinus) is listed as the national marine mammal.
  Fact 0.86: In Nepal, the Cow (Bos taurus indicus) is listed as the national animal.
  Fact 0.87: In Nepal, the Danphe (Lophophorus impejanus) is listed as the national bird.
  Fact 0.88: In the Netherlands, the Lion (Panthera leo) is listed as the heraldic symbol.
  Fact 0.89: In the Netherlands, the Black-tailed godwit (Limosa limosa) is listed as the national bird.
  Fact 0.90: In New Zealand, the Kiwi (Apteryx sp.) is listed.
  Fact 0.91: In North Korea, the Chollima is listed as a mythical animal.
  Fact 0.92: In Norway, the Lion (Panthera leo) is listed as the royal national animal.
  Fact 0.93: In Norway, the White-throated dipper (Cinclus cinclus) is listed as the national bird.
  Fact 0.94: In Norway, the Fjord horse (Equus ferus caballus) is listed as the national horse.
  Fact 0.95: In Oman, the Arabian oryx (Oryx leucoryx) is listed as the national animal.
  Fact 0.96: In Pakistan, the Markhor (Capra falconeri) is listed as the National animal.
  Fact 0.97: In Pakistan, the Indus river dolphin (Platanista indicus minor minor) is listed as the National aquatic mammal.
  Fact 0.98: In Pakistan, the Chukar (Alectoris chukar) is listed as the National bird.
  Fact 0.99: In Pakistan, the Indus crocodile (Crocodylus palustris) is listed as the National reptile.
  Fact 0.100: In Pakistan, the Shaheen falcon (Falco peregrinus peregrinator) is listed as the Heritage bird.
  Fact 0.101: In Pakistan, the Snow leopard (Panthera uncia) is listed as the National predator.
  Fact 0.102: In Panama, the Harpy eagle (Harpia harpyja) is listed.
  Fact 0.103: In Papua New Guinea, the Dugong (Dugong dugon) is listed as the national marine mammal.
  Fact 0.104: In Peru, the Vicuña (Vicugna vicugna) is listed as the national animal.
  Fact 0.105: In the Philippines, the Carabao (Bubalus bubalis) is listed as the national animal.
  Fact 0.106: In the Philippines, the Philippine eagle (Pithecophaga jefferyi) is listed as the national bird.
  Fact 0.107: In Poland, the White-tailed eagle (Haliaeetus albicilla) is listed as the national bird.
  Fact 0.108: In Poland, the White stork (Ciconia ciconia) is listed as the national bird.
  Fact 0.109: In Poland, the European bison (Bison bonasus) is listed as the national animal.
  Fact 0.110: In Qatar, the Arabian oryx (Oryx leucoryx) is listed.
  Fact 0.111: In Romania, the Lynx (Lynx lynx) is listed.
  Fact 0.112: In Russia, the Eurasian brown bear (Ursus arctos arctos) is listed.
  Fact 0.113: In Rwanda, the Leopard (Panthera pardus) is listed.
  Fact 0.114: In Saint Vincent and the Grenadines, the Saint Vincent amazon (Amazona guildingii) is listed.
  Fact 0.115: In Serbia, the Wolf (Canis lupus lupus) is listed.
  Fact 0.116: In Serbia, the White eagle is listed as a mythical animal.
  Fact 0.117: In Serbia, the Lynx (Lynx lynx) is listed.
  Fact 0.118: In Somalia, the Leopard (Panthera pardus) is listed as the national animal.
  Fact 0.119: In South Africa, the Springbok (Antidorcas marsupialis) is listed as the national animal.
  Fact 0.120: In South Africa, the Blue crane (Anthropoides paradiseus) is listed as the national bird.
  Fact 0.121: In South Africa, the Galjoen (Dichistius capensis) is listed as the national fish.
  Fact 0.122: In South Korea, the Siberian tiger (Panthera tigris altaica) is listed.
  Fact 0.123: In South Korea, the Korean magpie (Pica (pica) sericea) is listed as the national bird.
  Fact 0.124: In Sri Lanka, the Sri Lankan junglefowl (Gallus lafayettii) is listed as the national bird.
  Fact 0.125: In Tanzania, the Giraffe (Giraffa camelopardalis) is listed.
  Fact 0.126: In Thailand, the Elephant (Elephas maximus) is listed.
  Fact 0.127: In the United Arab Emirates, the Arabian oryx (Oryx leucoryx) is listed as the national animal.
  Fact 0.128: In the United Arab Emirates, the Peregrine falcon (Falco peregrinus) is listed as the national bird.
  Fact 0.129: In the United Kingdom, the Lion (Panthera leo) is listed for the United Kingdom, England, Scotland and Wales.
  Fact 0.130: In the United Kingdom, the Bulldog (Canis lupus familiaris) is listed as Britain, called the British Bulldog.
  Fact 0.131: In the United Kingdom, the Red kite (Milvus milvus) is listed for Wales.
  Fact 0.132: In the United Kingdom, the Unicorn is listed as a mythical animal for Scotland.
  Fact 0.133: In the United Kingdom, the Y Ddraig Goch (Welsh Dragon) is listed as a mythical animal for Wales.
  Fact 0.134: In the United States, the Bald eagle (Haliaeetus leucocephalus) is listed as the national bird.
  Fact 0.135: In the United States, the American bison (Bison bison) is listed as the national mammal.

Retrieved items:
<item 1>
List of national animals
List of National Animals - wikipedia List of National Animals Jump to : navigation , search This article is incomplete . Please help to improve it , or discuss
the issue on the talk page . ( June 2017 ) This is a list of national animals . National animals Country Name of animal Scientific name Pictures Ref . Algeria
Fennec fox Vulpes zerda Angola Red - crested turaco ( national bird ) Tauraco erythrolophus Anguilla Zenaida dove Zenaida aurita Antigua and Barbuda Fallow deer
( national animal ) Dama dama Frigate ( national bird ) Fregata magnificens Hawksbill turtle ( national sea creature ) Eretmochelys imbricata Argentina Rufous
hornero Furnarius rufus Australia Red kangaroo ( national animal ) Macropus rufus Emu ( national bird ) Dromaius novaehollandiae Austria Black eagle Ictinaetus
malaiensis Azerbaijan Karabakh horse Equus ferus caballus Bangladesh Royal Bengal tiger ( national animal ) Panthera tigris tigris Magpie robin ( national bird
) Copsychus saularis Ilish ( national fish )
</item 1>

<item 2>
The excerpt is from an incomplete Wikipedia list of countries’ national animals, birds, and other symbolic creatures, often with scientific names and
references. Examples include Algeria’s fennec fox, Australia’s red kangaroo and emu, Bangladesh’s Bengal tiger and magpie robin, Tanzania’s giraffe, and the
UAE’s Arabian oryx and peregrine falcon. It also lists the UK’s lion, bulldog, red kite, unicorn, and Welsh dragon, and the United States’
</item 2>

<item 3>
List of national animals
bird ) Gallus lafayettii Tanzania Giraffe Giraffa camelopardalis Thailand Elephant Elephas maximus United Arab Emirates Arabian oryx ( national animal ) Oryx
leucoryx Peregrine falcon ( national bird ) Falco peregrinus United Kingdom Lion ( United Kingdom , England , Scotland , Wales ) Panthera leo Bulldog ( Britain
as `` British Bulldog '' ) Canis lupus familiaris Red kite ( Wales ) Milvus milvus Unicorn ( Scotland ) Mythical Y Ddraig Goch ( Welsh Dragon Wales ) Mythical
United States Bald eagle ( national bird ) Haliaeetus leucocephalus American bison ( national mammal ) Bison bison See also Animals portal Lists portal State
animal List of national birds Animals as heraldic charges Floral emblem National personification References Jump up ^ `` The Fennec Fox : Algeria 's National
Animal '' . algeria.com . Retrieved 2013 - 03 - 21 . Jump up ^ `` THE NATIONAL BIRD OF ANGOLA '' . BIRDS ANGOLA . Retrieved 12 April 2016 .
</item 3>

<item 4>
A country-by-country catalogue pairs national animals, birds, fish and other symbols with scientific names. Examples include Bangladesh’s magpie-robin and
ilish; Belarus’s bison; China’s panda, dragon and crane; India’s tiger, peafowl, cobra, river dolphin and elephant; Mexico’s eagle, Xoloitzcuintli, jaguar and
vaquita; and Pakistan’s markhor, dolphin, chukar, crocodile, falcon and snow leopard. Other
</item 4>

<item 5>
An incomplete Wikipedia-style catalogue lists countries’ national animals, birds, fish, and other symbolic creatures, often with scientific names, alongside
extensive citations from government and tourism sites, news, books, and reference works. Examples include India’s tiger and peafowl, China’s panda and crane,
Mexico’s eagle and vaquita, and Pakistan’s markhor. The material also includes related national-symbol references and separate catalogues of national, regional,
and historical anthems, including disputed territories and former states
</item 5>

<item 6>
A bibliography on national animals and symbols cites a Philippine proclamation, references identifying Poland’s white-tailed eagle/White Eagle, white stork, and
European bison, a book on Qatar, a Romanian article about the lynx, information on Birobidzhan, and a guide to Rwanda’s state symbols.
</item 6>

<item 7>
List of national animals
Vicugna vicugna Philippines Carabao ( national animal ) Bubalus bubalis Philippine eagle ( national bird ) Pithecophaga jefferyi Poland White - tailed eagle (
national bird ) Haliaeetus albicilla White stork ( national bird ) Ciconia ciconia European bison ( national animal ) Bison bonasus Qatar Arabian oryx Oryx
leucoryx Romania Lynx Lynx lynx Russia Eurasian brown bear Ursus arctos arctos Rwanda Leopard Panthera pardus Saint Vincent and the Grenadines Saint Vincent
amazon Amazona guildingii Serbia Wolf Canis lupus lupus White eagle Mythical Lynx Lynx lynx Somalia Leopard ( national animal ) Panthera pardus South Africa
Springbok ( national animal ) Antidorcas marsupialis Blue crane ( national bird ) Anthropoides paradiseus Galjoen ( national fish ) Dichistius capensis South
Korea Siberian tiger Panthera tigris altaica Korean magpie ( national bird ) Pica ( pica ) sericea Sri Lanka Sri Lankan junglefowl ( national bird ) Gallus
lafayettii Tanzania Giraffe Giraffa camelopardalis Thailand Elephant
</item 7>

<item 8>
List of national animals
versicolor Carp ( national fish ) Cyprinus carpio Latvia White wagtail ( national bird ) Motacilla alba Two - spotted ladybird ( national insect ) Adalia
bipunctata Lithuania White stork Ciconia ciconia Macedonia Lion ( in Macedonian heraldry ) Panthera leo Madagascar Zebu ( in Malagasy heraldry ) Bos primigenius
indicus Ring - tailed lemur ( national animal ) Lemur catta Malaysia Malayan tiger ( national animal ) Panthera tigris jacksoni Malta Pharaoh Hound ( national
animal ) Canis lupus familiaris ( 1 ) Moldova Aurochs ( national animal ) Bos primigenius Mauritius Dodo Raphus cucullatus Mexico Golden eagle ( national animal
) Aquila chrysaetos Xoloitzcuintli ( national dog ) Canis lupus familiaris Jaguar ( national mammal ) Panthera onca Grasshopper ( national arthropod )
Sphenarium purpurascens Vaquita ( national marine mammal ) Phocoena sinus Nepal Cow ( national animal ) Bos taurus indicus Danphe ( national bird ) Lophophorus
impejanus Netherlands Lion ( heraldic symbol
</item 8>

<item 9>
List of national animals
Danphe ( national bird ) Lophophorus impejanus Netherlands Lion ( heraldic symbol ) Panthera leo Black - tailed godwit ( national bird ) Limosa limosa New
Zealand Kiwi Apteryx sp . North Korea Chollima Mythical Norway Lion ( royal national animal ) Panthera leo White - throated dipper ( national bird ) Cinclus
cinclus Fjord horse ( national horse ) Equus ferus caballus Oman Arabian oryx ( national animal ) Oryx leucoryx Pakistan Markhor ( National animal ) Capra
falconeri Indus river dolphin ( National aquatic mammal ) Platanista indicus minor minor Chukar ( National bird ) Alectoris chukar Indus crocodile ( National
reptile ) Crocodylus palustris Shaheen falcon ( Heritage bird ) Falco peregrinus peregrinator Snow leopard ( National predator ) Panthera uncia Panama Harpy
eagle Harpia harpyja Papua New Guinea Dugong ( national marine mammal ) Dugong dugon Peru Vicuña ( national animal ) Vicugna vicugna Philippines Carabao (
national animal ) Bubalus bubalis Philippine
</item 9>

<item 10>
List of national animals
Magpie robin ( national bird ) Copsychus saularis Ilish ( national fish ) Tenualosa ilisha Belarus European bison Bison bonasus Belgium Lion ( heraldic Leo
Belgicus ) Panthera leo Belize Baird 's tapir ( national animal ) Tapirus bairdii Keel - billed toucan ( national bird ) Ramphastos sulfuratus Bhutan Druk
Mythical Takin Budorcas taxicolor Brazil Rufous - bellied thrush Turdus rufiventris Cambodia Kouprey Bos sauveli Canada North American beaver ( sovereignty
animal symbol ) Castor canadensis Canadian horse ( national horse ) Equus ferus caballus China Giant panda ( national animal ) Ailuropoda melanoleuca Chinese
dragon ( national animal ) Mythical Red - crowned crane ( national bird ) Grus japonensis Democratic Republic of the Congo Okapi Okapia johnstoni Colombia
Andean condor Vultur gryphus Costa Rica Yigüirro ( national bird ) Turdus grayi White - tailed deer ( national animal ) Odocoileus virginianus West Indian
manatee ( national aquatic animal ) Trichechus manatus Croatia Pine
</item 10>
=== END ITEM id=81bb4384859bc55a ===

=== ITEM id=3d4470c78fdbd975 ===
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
Villanova has won 4 NCAA Men 's Cross Country Team Championships covering 1966 , 1967 , 1968 and 1970 . NCAA Individual Cross Country Champions Victor Zwolak
1963 Patrick Tiernan 2016 Swimming and Diving ( edit ) Main article : Villanova Swimming and Diving In its history , Villanova Swimming and Diving has produced
several Olympians . Most recently Maddy Crippen competed in the 400m individual medley at the 2000 Olympics . At the 2008 Olympics , Villanova was represented
by Kristina Lennox - Silva . Lennox competed for Puerto Rico and swam the 400m freestyle and 200m butterfly . NCAA team Championships ( edit ) Villanova has 21
NCAA team national championships .
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
Track and field ( edit ) Villanova University 's Track and field team has a history of athletic success that has spanned from championships in both incarnations
of the Big East Conference to NCAA championships . The men 's team has produced 69 NCAA championships , 36 indoor and 33 outdoor . The team has had 8 NCAA team
championships -- 4 cross country , 3 indoor ( ' 68 , ' 71 , ' 79 ) , 1 outdoor ( ' 57 ) . Prior to NCAA sponsoring Indoor T&F in 1965 , Villanova won 6 ICAAAA
Indoor T&F Championships ( ' 57 , ' 58 , ' 60 , ' 62 , ' 63 , ' 64 ) . Villanova has produced 28 athletes who have made appearances in the Olympics , 10 of whom
have medaled ( 7 Gold medals , 3 Silver medals ) . The men 's team has also won 112 Penn Relays Championships , which stands as the most wins by any school .
The men 's current coaches include head coach , Marcus O'Sullivan , and assistant head coach , Anthony Williams .
</item 3>

<item 4>
Villanova Wildcats
During the 2017 - 18 season , the team played its entire home schedule that season at the Wells Fargo Center following the reconstruction of the Pavilion
scheduled to be completed in time for the 2018 - 19 school year . Wright has taken the Wildcats to 12 postseason appearances including 7 consecutive trips (
2005 -- 2011 ) and 5 consecutive trips ( 2013 - 2017 ) to the NCAA Tournament . The 2011 - 12 team struggled and missed postseason but Coach Wright brought a
surprising young team back to the 2013 NCAA Tournament . Villanova has appeared in the NIT 17 times , winning in 1994 , and won the Big East Tournament in 1995
, 2015 and 2017 . The Wildcats have appeared in the NCAA Men 's Tournament 37 times , the 8th highest total in NCAA history have reached the Elite Eight 13
times . They have been to the Final Four of the NCAA Division I Men 's Basketball Tournament on five occasions . The 1939 team coached by Al Severance reached
the inaugural NCAA Final Four played at the Palestra for the 1939 NCAA Basketball Tournament .
</item 4>

<item 5>
Villanova Wildcats
Villanova was the Big East regular season and tournament champions in both 1985 - 86 and 1986 - 87 behind All - American Shelly Pennefather . During the 2008 -
09 season , Coach Perretta led the Wildcats to the NCAA Tournament for the ninth time in his Villanova career . This marked the ninth postseason appearance in
the previous 10 years for the Wildcats . After playing in the 2000 WNIT , Villanova went to four consecutive NCAA Tournaments from 2001 to 2004 before playing
in the WNIT in 2005 , 2006 and 2008 . The 2002 - 03 season was a special one for the Wildcats . In addition to an impressive 28 - 6 overall record and a 12 - 4
Big East Conference mark , the Wildcats won their third Big East Conference Tournament championship and first since 1987 with an improbable 52 - 48 upset over
previously unbeaten Connecticut . The win snapped Connecticut 's then NCAA Division I women 's basketball record of 70 - straight victories and ended
Connecticut 's streak of nine - straight conference tournament championships .
</item 5>

<item 6>
Villanova Wildcats
NCAA team Championships ( edit ) Villanova has 21 NCAA team national championships . Men 's ( 12 ) Basketball ( 3 ) : 1985 , 2016 , 2018 Cross Country ( 4 ) :
1966 , 1967 , 1968 , 1970 Football ( 1 ) : 2009 Indoor Track & Field ( 3 ) : 1968 , 1971 , 1979 Outdoor Track & Field ( 1 ) : 1957 Women 's ( 9 ) Cross Country
( 9 ) : 1989 , 1990 , 1991 , 1992 , 1993 , 1994 , 1998 , 2009 , 2010 see also : Big East Conference NCAA team championships List of NCAA schools with the most
NCAA Division I championships Conference realignment ( edit ) In December 2012 the presidents of seven Big East schools which do not sponsor FBS football ( also
sometimes referred to as the `` Catholic 7 '' ) , including Villanova , voted unanimously to leave the Big East and form a new conference .
</item 6>

<item 7>
Villanova Wildcats
For the 2017 -- 18 season , they will play their home games at Jake Nevin Field House on the Villanova campus while the future Finneran Pavilion is undergoing
renovations . The Women 's basketball program started in 1969 and played under CIAW and AIAW through 1981 - 82 season before moving to NCAA Division I. In the
81 - 82 season , the Wildcats posted an outstanding 29 - 4 record and reached the Final Four of the 1982 AIAW Women 's Basketball Tournament before losing to
eventual National Champion , Rutgers . Through the 2009 - 10 season , the Villanova Women have a 669 - 413 alltime record since the program started in 1969 .
Previous head coaches include Liz Cawley ( 1969 -- 70 ) , Jane Sefranek - Stoltz ( 1970 -- 75 ) and Joan King ( 1975 -- 78 ) . The 1982 - 83 season was the
initial season for Villanova Women 's basketball to play NCAA Division I and they joined the Big East Conference that same season .
</item 7>

<item 8>
Villanova Wildcats
Coach Michael Corrado has taken the team to its only two NCAA appearances in history during his 6 - year tenure and the school is poised for greater success in
upcoming seasons . Villanova rose as high as # 7 ranking and qualified for an at - large bid in the 2011 NCAA Tournament where they lost to the University of
Denver in the first round . During the regular season , the Wildcats drew over 6500 fans to a Big East match - up with Syracuse setting a new on - campus record
for Lacrosse . Brian Karalunas became the first USILA First Team All - American in Villanova history . Through 2009 , Villanova men 's lacrosse was a member of
the Colonial Athletic Association and won the CAA tournament for their first title in 2009 . The fourth - seeded Wildcats were the lowest - seeded championship
team in conference history . That season , Villanova also made their first NCAA tournament appearance , where they were routed in the first round by first -
seeded Virginia , 18 -- 6 . The 2004 team finished 13 - 3 playing against the stiff competition of the CAA .
</item 8>

<item 9>
Villanova Wildcats
617 ) record . Since moving to I - AA , Villanova has made 9 appearances in the I - AA / FCS playoffs making trips to the NCAA Semifinal in 2002 and 2010 and
winning the 2009 FCS National Championship . The Wildcats played in the NCAA Division I Football Championship game on December 18 , 2009 against the Montana
Grizzlies . The Wildcats beat the Grizzlies 23 -- 21 to win their first NCAA Division I - AA National Championship . Villanova played in bowl games following
the 1936 , 1947 , 1948 , 1961 , and 1962 college football seasons . After 87 seasons , the Board of Trustees voted to drop Football in April 1981 due to
financial losses and poor attendance . In April , 1984 , the Board of Trustees agreed to reinstate Football at a lower level as a non-scholarship program (
Division I schools could field a non-scholarship Division III program at the time ) and named Andy Talley as the head coach , with first Division III practice
that fall and first .
</item 9>

<item 10>
Villanova Wildcats
The Wildcat squad remains the lowest overall seed in tournament history to win the championship , and their overall team shooting percentage of 78.6 % remains
an NCAA tournament record for a single game . This Championship Game is often cited among the greatest upsets in college basketball history . Ed Pinckney , who
shot 5 - of - 7 and had 16 points in the game , was named the NCAA Tournament 's Most Outstanding Player . On April 4 , 2016 , the Wildcats won their second
NCAA Championship in Men 's Basketball , the first time in 31 years . They beat the North Carolina Tar Heels 77 - 74 off a buzzer - beater shot by Kris Jenkins
with 0.5 seconds remaining in regulation . Two years later , on April 2 , 2018 , Villanova earned its third national championship beating the Michigan
Wolverines 79 - 62 . Women 's Basketball ( edit ) Main article : Villanova Wildcats women 's basketball 600th win The Villanova Wildcats women 's basketball
team are coached by Harry Perretta , now serving his 34th year as head coach of the team .
</item 10>
=== END ITEM id=3d4470c78fdbd975 ===

=== ITEM id=1cdcdc0f2b5ff84b ===
Gold sentences and facts:
Sentence 0: In every High Court , there is a Chief Justice and many other judges whose number is defined by the President of India .
  Fact 0.0: Every High Court has a Chief Justice.
  Fact 0.1: Every High Court has many other judges.
  Fact 0.2: The number of other judges in a High Court is defined by the President of India.

Retrieved items:
<item 1>
List of Justices of the Supreme Court of Sri Lanka
Imam Central 9 January 2009 -- 20 February 2013 -- 1,503 -- Rajapaksa Judge of the Court of Appeal R.K.S. Suresh Chandra 17 June 2010 -- -- -- Rajapaksa Sathya
Hettige 10 July 2011 -- 18 April 2014 -- 1,013 -- Rajapaksa University of Colombo President of the Court of Appeal of Sri Lanka Priyasath Dep † 10 July 2011 --
present 02 March 2017 - present -- 7003248200000000000 ♠ 2,482 -- Rajapaksa University of Colombo International Institute of Social Studies Solicitor General of
Sri Lanka Eva Wansundera † North Western 7 July 2012 -- present -- 7003211900000000000 ♠ 2,
</item 1>

<item 2>
Indian High Courts Act 1861
Qualifications for the Judges ( a ) He should be a citizen of India . ( b ) He should have been ( I ) a judge for 10 years of Subordinate court under the
Judicial Service of the State or ( ii ) an Advocate for 10 years in a High Courts in India ( Article 217 ) . Tenure : Originally the age of the retirement of
the judges of the High Courts was fixed at 60 but it was raised to 62 in 1963 according to the 15th amendment of the Constitution . Removal of the Judges : A
judge may leave his office by resigning . He will send his letter of resignation to the President . His office would be considered to have been vacated if he is
appointed as a judge of the Supreme Court or is transferred to some other High Court . A judge of a High Court may also be removed like a judge of the Supreme
Court . A judge of High Court may be removed by the President if the Parliament passes a motion against him by an absolute majority and 2 / 3rd majority of the
members present and voting , both the Houses sitting separately .
</item 2>

<item 3>
List of Justices of the Supreme Court of Sri Lanka
182 -- Sirisena Sri Lanka Law College University of Exeter Judge of the Court of Appeal Kankanithanthri Chitrasiri † 3 December 2015 -- present --
7002875000000000000 ♠ 875 -- Sirisena University of Sri Lanka Queen Mary University of London Judge of the Court of Appeal Nalin Perera † 3 March 2016 --
present -- 7002784000000000000 ♠ 784 -- Sirisena Judge of the Court of Appeal Prasanna Sujeewa Jayawardena † 16 June 2016 -- present -- 7002679000000000000 ♠
679 -- Sirisena University of Colombo Judge of the Court of Appeal See also ( edit ) List of Chief Justices of Sri Lanka List of Justices of the Supreme Court
of Sri Lanka by court composition Notes ( edit ) Jump up ^ Provisional Chief Justice Jump up ^ On leave ( 1 January - 28 April 1871 ) Jump up ^ Acting Chief
Justice ( 1 - 10 October 1879 ,
</item 3>

<item 4>
Indian High Courts Act 1861
Salary : The pay of the Chief Justice of a High Court is rupees 90,000 / - per month and that of the other judges is rupees 85,000 / - per month . Powers and
Functions Original Jurisdiction : The original jurisdiction of the High Court is restricted . ( a ) Every High Court under Article 226 is empowered to issue
writs , orders , directions including writs in the nature of Habeas Corpus , Mandamus , Prohibition , Quo - warranto and Certiorari or any of them to any person
or authority with in its territory for the enforcement of the Fundamental Rights and for any other purpose . ( b ) The original jurisdiction of High Court
extends to matters of admiralty , matrimonial , contempt of court and cases ordered to be transferred to High Court by lower court . ( c ) The High Courts of
Mumbai , Kolkata and Chennai have original jurisdiction on hearing straightway cases involving the Christians and Parsies . ( d ) The High Courts of Mumbai ,
Kolkata and Chennai exercise original civil jurisdiction when the amount involved is more than two thousand rupees .
</item 4>

<item 5>
Indian High Courts Act 1861
It exercises the power of superintendence and control over all courts and tribunals throughout the territory except the military tribunals . Judicial tenure and
seniority ( edit ) Under § 5 of the Act , judges served at the pleasure of Her Majesty . The chief justice had precedence over judges , whereas judges had
seniority based on appointment . Under § 14 , the chief justice had the power to form divisions of two or more judges to hear cases . Jurisdiction ( edit )
Under article 9 of the Act , each High Court had `` all such powers and authority for and in relation to the administration of justice '' including original and
appellate jurisdiction over civil , criminal , admiralty , vice-admiralty , testamentary , intestate , and matrimonial matters . References ( edit ) Jump up ^
`` Indian High Courts Act 1861 '' . GKToday. 2011 - 10 - 17 . Retrieved 2017 - 03 - 20 . Jump up ^ Criminal Justice India Series : Punjab , 2002 . Allied
Publishers . 2002 . p. 233 .
</item 5>

<item 6>
Indian High Courts Act 1861
Appellate Jurisdiction : The appellate jurisdiction of the High Courts extends so : ( a ) The High Court can hear appeals in civil cases if the amount involved
in the case is at least Rs. 5000 . ( b ) The High Court in criminal cases hears the appeal in which the accused has been sentenced to four years imprisonment by
the Sessions Judge . v ( c ) The death sentence awarded by Sessions Judge is subject to approval by the High Court . ( d ) The High Court hear the cases
involving interpretation of the Constitution or Law . ( e ) The High Court hears the cases on income tax , sales tax etc . Power of Judicial Review : The States
High Courts like the Supreme Court has the power of Judicial Review . A High Court has the power to strike down any law of the State or any order of the
executive if it violates any provision of the constitution or curtails or takes any of the Fundamental Rights of the people . Administrative and Supervisory
Power : The State High Court performs many administrative functions within its Territorial Jurisdiction .
</item 6>

<item 7>
Indian High Courts Act 1861
The High Court of a State is the highest court of the State and all other courts of the State work under it . Normally there is one High Court in every State
but there can be only one High Court for two or more States as well , according to the constitution . There is one High Court at Chandigarh for Punjab , Haryana
and Union Territory of Chandigarh . Similarly there is one High Court at Guwahati which serves Assam , Arunachal Pradesh , Mizoram and Nagaland . Composition :
In every High Court , there is a Chief Justice and many other judges whose number is defined by the President of India . Appointment of the Judges : The Chief
Justice of a High Court is appointed by the President with the consultation of the Chief Justice of the Supreme Court and the Governor of the State . The other
judges are appointed by the will of President , Governor and the Chief Justice of High Court . Qualifications for the Judges ( a ) He should be a citizen of
India .
</item 7>

<item 8>
List of Justices of the Supreme Court of Sri Lanka
Justice of the Straits Settlements 35 George Henry Withers -- -- -- 36 Dodwell Browne -- -- -- 37 Fredrick Charles Moncrieff 4 April 1900 -- -- -- 38 Henry
Lorentz Wendt 15 October 1901 -- 28 November 1910 -- -- 39 John Page Middleton 28 January 1902 -- 1912 -- -- Puisine Judge of the Supreme Court of Cyprus 40
Charles Layard Western 26 April 1902 -- 18 June 1906 26 April 1902 -- 18 June 1906 1,514 1,514 41 Joseph Richard Grenier 28 November 1910 -- 1912 -- -- 42
Alexander Wood Renton 3 September 1905 -- 1918 22 August 1914 - 1918 -- 43 Alfred George Lascelles England 1 May 1911 -- 1914 1 May 1911 -- 1914 -- 44 Joseph
Turner Hutchinson England 23 October 1906 -- 1 May 1911 23 October 1906 -- 1 May 1911 1,
</item 8>

<item 9>
Indian High Courts Act 1861
Contents ( hide ) 1 Abolishing Existing Courts 2 Qualifications of High Court Judges 3 Judicial Tenure and Seniority 4 Jurisdiction 5 References Abolishing
existing Courts ( edit ) The Act abolished the Supreme Courts at Calcutta , Madras , and Bombay ; the Sadar Diwani Adalat and the Sadar Nizamat Adalat at
Calcutta ; Sadar Adalat and Faujdari Adalat at Madras ; Sadar Diwani Adalat and Faujdari Adalat at Bombay ( § 8 ) . Qualifications of High Court judges ( edit )
Each High Court could consist of a chief justice and up to 15 judges . Under § 3 of the Act , judges could be selected from barristers ( with 5 years of
experience ) , civil servants ( with 10 years of experience including 3 years as a zillah judge ) , judges of small cause courts or sudder ameen ( with 5 years
of experience ) , or pleaders of lander courts or High Courts ( with 05 years of experience ) .
</item 9>

<item 10>
Justice of the peace
Majesty 's Courts and Tribunals Service Judges ' Council Civil and family courts ( show ) Supreme Court of the United Kingdom Privy Council Court of Appeal
Master of the Rolls Court of Appeal judge High Court of Justice President of the Queen 's Bench President of the Family Division Chancellor of the High Court
High Court judge County courts List of county court venues County Court Business Centre District judge Family Court Court of Protection Court of Chivalry
Criminal courts ( show ) Supreme Court of the United Kingdom Court of Appeal Lord Chief Justice Court of Appeal judge High Court of Justice President of the
Queen 's Bench High Court judge Crown Court List of Crown Court venues Circuit judge Recorder Magistrates ' courts District judge Justice of the Peace / Lay
magistrates Criminal prosecution ( show ) Attorney General Director of Public Prosecutions Crown Prosecution Service Legal profession ( show ) Bar Council
Barrister Law Society of England and Wales Solicitor Solicitor Advocate Main article : Magistrate ( England and Wales ) A magistrates ' court in England and
Wales is composed of a bench of ( usually three ) JPs or magistrates , who dispense summary justice : that is they decide on offences which carry up to six
months in prison ,
</item 10>
=== END ITEM id=1cdcdc0f2b5ff84b ===

=== ITEM id=53c6f7fd6c5c844a ===
Gold sentences and facts:
Sentence 0: Transcription occurs in the cell nucleus , where the DNA is held . The DNA structure of the cell is made up of two helixes made up of sugar and phosphate held together by hydrogen bonds between the bases of opposite strands . The sugar and the phosphate in each strand are joined together by stronger phosphodiester covalent bonds . The DNA is `` unzipped '' ( disruption of hydrogen bonds between different single strands ) by the enzyme helicase , leaving the single nucleotide chain open to be copied . RNA polymerase reads the DNA strand from the 3 - prime ( 3 ' ) end to the 5 - prime ( 5 ' ) end , while it synthesizes a single strand of messenger RNA in the 5 ' - to - 3 ' direction . The general RNA structure is very similar to the DNA structure , but in RNA the nucleotide uracil takes the place that thymine occupies in DNA . The single strand of mRNA leaves the nucleus through nuclear pores , and migrates into the cytoplasm .
  Fact 0.0: Transcription occurs in the cell nucleus.
  Fact 0.1: The DNA is held in the cell nucleus.
  Fact 0.2: The DNA structure of the cell is made up of two helixes.
  Fact 0.3: Each helix of the DNA structure is made up of sugar and phosphate.
  Fact 0.4: The two helixes of the DNA structure are held together by hydrogen bonds between the bases of opposite strands.
  Fact 0.5: The sugar and the phosphate in each strand of DNA are joined together by phosphodiester covalent bonds.
  Fact 0.6: The phosphodiester covalent bonds are stronger than the hydrogen bonds.
  Fact 0.7: The DNA is "unzipped" by the enzyme helicase.
  Fact 0.8: The unzipping of the DNA is disruption of hydrogen bonds between different single strands.
  Fact 0.9: Unzipping the DNA leaves the single nucleotide chain open to be copied.
  Fact 0.10: RNA polymerase reads the DNA strand from the 3' end to the 5' end.
  Fact 0.11: While reading the DNA strand, RNA polymerase synthesizes a single strand of messenger RNA.
  Fact 0.12: RNA polymerase synthesizes the messenger RNA in the 5' to 3' direction.
  Fact 0.13: The general RNA structure is very similar to the DNA structure.
  Fact 0.14: In RNA, the nucleotide uracil takes the place that thymine occupies in DNA.
  Fact 0.15: The single strand of messenger RNA leaves the nucleus through nuclear pores.
  Fact 0.16: The single strand of messenger RNA migrates into the cytoplasm.

Retrieved items:
<item 1>
Protein biosynthesis
The first product of transcription differs in prokaryotic cells from that of eukaryotic cells , as in prokaryotic cells the product is mRNA , which needs no
post-transcriptional modification , whereas , in eukaryotic cells , the first product is called primary transcript , that needs post-transcriptional
modification ( capping with 7 - methyl - guanosine , tailing with a poly A tail ) to give hnRNA ( heterogeneous nuclear RNA ) . hnRNA then undergoes splicing of
introns ( noncoding parts of the gene ) via spliceosomes to produce the final mRNA . Translation ( edit ) Main article : Translation ( biology ) Diagram showing
the process of translation Diagram showing the translation of mRNA and the synthesis of proteins by a ribosome Phenomena of amino acid assembly from RNA . The
synthesis of proteins from RNA is known as translation . In eukaryotes , translation occurs in the cytoplasm , where the ribosomes are located . Ribosomes are
made of a small and large subunit that surround the mRNA .
</item 1>

<item 2>
Protein biosynthesis
Ribosomes are made of a small and large subunit that surround the mRNA . In translation , messenger RNA ( mRNA ) is decoded to produce a specific polypeptide
according to the rules specified by the trinucleotide genetic code . This uses an mRNA sequence as a template to guide the synthesis of a chain of amino acids
that form a protein . Translation proceeds in four phases : activation , initiation , elongation , and termination ( all describing the growth of the amino acid
chain , or polypeptide that is the product of translation ) . In activation , the correct amino acid ( AA ) is joined to the correct transfer RNA ( tRNA ) .
While this is not , in the technical sense , a step in translation , it is required for translation to proceed . The AA is joined by its carboxyl group to the 3
' OH of the tRNA by an ester bond . When the tRNA has an amino acid linked to it , it is termed `` charged '' . Initiation involves the small subunit of the
ribosome binding to 5 ' end of mRNA with the help of initiation factors ( IF ) , other proteins that assist the process .
</item 2>

<item 3>
Protein biosynthesis
at 22 : 32 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the
Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 3>

<item 4>
Protein biosynthesis
Protein biosynthesis - wikipedia Protein biosynthesis Jump to : navigation , search RNA is transcribed in the nucleus ; once completely processed , it is
transported to the cytoplasm and translated by the ribosome ( shown in very pale grey behind the tRNA ) . Protein synthesis is the process whereby biological
cells generate new proteins ; it is balanced by the loss of cellular proteins via degradation or export . Translation , the assembly of amino acids by ribosomes
, is an essential part of the biosynthetic pathway , along with generation of messenger RNA ( mRNA ) , aminoacylation of transfer RNA ( tRNA ) ,
co-translational transport , and post-translational modification . Protein biosynthesis is strictly regulated at multiple steps . They are principally during
transcription ( phenomena of RNA synthesis from DNA template ) and translation ( phenomena of amino acid assembly from RNA ) . The cistron DNA is transcribed
into the first of a series of RNA intermediates . The last version is used as a template in synthesis of a polypeptide chain . Protein will often be synthesized
directly from genes by translating mRNA .
</item 4>

<item 5>
Protein biosynthesis
The termini and side - chains of the polypeptide may be subjected to post-translational modification . These modifications may be required for correct cellular
localisation or the natural function of the protein . During and after synthesis , polypeptide chains often fold to assume , so called , native secondary and
tertiary structures . This is known as protein folding and is typically required for the natural function of the protein . See also ( edit ) Central dogma of
molecular biology Cistron Gene expression Genetic code Operon Peptide synthesis Protein production References ( edit ) Jump up ^ Kafri M , Metzl - Raz E , Jona
G , Barkai N. 2016 . The Cost of Protein Production . Cell Rep 14 : 22 -- 31 . https://dx.doi.org/10.1016/j.celrep.2015.12.015 ^ Jump up to : Alberts , Bruce (
2002 ) . Molecular biology of the cell . New York : Garland Science . p. 760 . ISBN 0 - 8153 - 3218 - 1 . Jump up ^ Alberts , Bruce .
</item 5>

<item 6>
Protein biosynthesis
This whole complex of processes is carried out by the ribosome , formed of two main chains of RNA , called ribosomal RNA ( rRNA ) , and more than 50 different
proteins . The ribosome latches onto the end of an mRNA molecule and moves along it , capturing loaded tRNA molecules and joining together their amino acids to
form a new protein chain . Protein biosynthesis , although very similar , is different for prokaryotes and eukaryotes . Contents ( hide ) 1 Transcription 2
Translation 3 Events during or following protein translation 4 See also 5 References 6 External links Transcription ( edit ) Main article : Transcription (
genetics ) Diagram showing the process of transcription Phenomena of RNA synthesis from DNA template In transcription an mRNA chain is generated , with one
strand of the DNA double helix in the genome as a template . This strand is called the template strand . Transcription can be divided into 3 stages : initiation
, elongation , and termination , each regulated by a large number of proteins such as transcription factors and coactivators that ensure that the correct gene
is transcribed . Transcription occurs in the cell nucleus , where the DNA is held .
</item 6>

<item 7>
Protein biosynthesis
Elongation occurs when the next aminoacyl - tRNA ( charged tRNA ) in line binds to the ribosome along with GTP and an elongation factor . Termination of the
polypeptide happens when the A site of the ribosome faces a stop codon ( UAA , UAG , or UGA ) . When this happens , no tRNA can recognize it , but releasing
factor can recognize nonsense codons and causes the release of the polypeptide chain . The capacity of disabling or inhibiting translation in protein
biosynthesis is used by some antibiotics such as anisomycin , cycloheximide , chloramphenicol , tetracycline , streptomycin , erythromycin , puromycin , etc .
Events during or following protein translation ( edit ) Main articles : Proteolysis , Posttranslational modification , and Protein folding Events that occur
during or following biosynthesis include proteolysis , post-translational modification and protein folding . Proteolysis may remove N - terminal , C - terminal
or internal amino - acid residues or peptides from the polypeptide .
</item 7>

<item 8>
Protein biosynthesis
Jump up ^ Alberts , Bruce . Molecular Biology of the Cell , 5e . New York : Garland Science , 2008 . External links ( edit ) Interactive Java simulation of
transcription initiation . From Center for Models of Life at the Niels Bohr Institute . A website focused primarily on Protein Synthesis . Protein primary
structure and posttranslational modifications General Peptide bond Protein biosynthesis Proteolysis Racemization N-O acyl shift N terminus Acetylation
Carbamylation Formylation Glycation Methylation Myristoylation ( Gly ) C terminus Amidation Glycosyl phosphatidylinositol ( GPI ) O - methylation Detyrosination
Single specific AAs Serine / Threonine Phosphorylation Dephosphorylation Glycosylation Methylidene - imidazolone ( MIO ) formation Tyrosine Phosphorylation
Dephosphorylation Sulfation Porphyrin ring linkage Adenylylation Flavin linkage Topaquinone ( TPQ ) formation Detyrosination Cysteine Palmitoylation Prenylation
Aspartate Succinimide formation Glutamate Carboxylation Methylation
</item 8>

<item 9>
Protein biosynthesis
Protein will often be synthesized directly from genes by translating mRNA . However , when a protein must be available on short notice or in large quantities ,
a protein precursor is produced . A proprotein is an inactive protein containing one or more inhibitory peptides that can be activated when the inhibitory
sequence is removed by proteolysis during posttranslational modification . A preprotein is a form that contains a signal sequence ( an N - terminal signal
peptide ) that specifies its insertion into or through membranes , i.e. , targets them for secretion . The signal peptide is cleaved off in the endoplasmic
reticulum . Preproproteins have both sequences ( inhibitory and signal ) still present . In protein synthesis , a succession of tRNA molecules charged with
appropriate amino acids are brought together with an mRNA molecule and matched up by base - pairing through the anti-codons of the tRNA with successive codons
of the mRNA . The amino acids are then linked together to extend the growing protein chain , and the tRNAs , no longer carrying amino acids , are released .
</item 9>

<item 10>
Protein biosynthesis
Transcription occurs in the cell nucleus , where the DNA is held . The DNA structure of the cell is made up of two helixes made up of sugar and phosphate held
together by hydrogen bonds between the bases of opposite strands . The sugar and the phosphate in each strand are joined together by stronger phosphodiester
covalent bonds . The DNA is `` unzipped '' ( disruption of hydrogen bonds between different single strands ) by the enzyme helicase , leaving the single
nucleotide chain open to be copied . RNA polymerase reads the DNA strand from the 3 - prime ( 3 ' ) end to the 5 - prime ( 5 ' ) end , while it synthesizes a
single strand of messenger RNA in the 5 ' - to - 3 ' direction . The general RNA structure is very similar to the DNA structure , but in RNA the nucleotide
uracil takes the place that thymine occupies in DNA . The single strand of mRNA leaves the nucleus through nuclear pores , and migrates into the cytoplasm .
</item 10>
=== END ITEM id=53c6f7fd6c5c844a ===
