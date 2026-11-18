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

=== ITEM id=f9d3109b1e530d42 ===
Gold sentences and facts:
Sentence 0: In 1918 , Firestone Tire and Rubber Company of Canada was incorporated in Hamilton , Ontario and in 1922 , the first Canadian - made tire rolled off the line on September 15 . During the 1920s , Firestone produced the Oldfield tire , named for racing driver Barney Oldfield .
  Fact 0.0: Firestone Tire and Rubber Company of Canada was incorporated in 1918.
  Fact 0.1: Firestone Tire and Rubber Company of Canada was incorporated in Hamilton, Ontario.
  Fact 0.2: The first Canadian-made tire rolled off the line in 1922.
  Fact 0.3: The first Canadian-made tire rolled off the line on September 15.
  Fact 0.4: During the 1920s, Firestone produced the Oldfield tire.
  Fact 0.5: The Oldfield tire was named for racing driver Barney Oldfield.
Sentence 1: Bridgestone / Firestone also produces tires under the Fuzion brand name . Fuzion is a tire manufactured in China for Bridgestone / Firestone as a budget replacement tire .
  Fact 1.0: Bridgestone/Firestone also produces tires under the Fuzion brand name.
  Fact 1.1: Fuzion is a tire.
  Fact 1.2: Fuzion is manufactured in China.
  Fact 1.3: Fuzion is manufactured for Bridgestone/Firestone.
  Fact 1.4: Fuzion is a budget replacement tire.

Retrieved items:
<item 1>
Firestone Tire and Rubber Company
ISBN 978 - 0 - 15 - 692620 - 1 . Jump up ^ `` A Global Dream '' . Bridgestone Americas . Retrieved 3 August 2013 . Jump up ^ `` History of Bridgestone /
Firestone '' . Bridgestone / Firestone Canada . 2009 . Archived from the original on 10 May 2010 . Jump up ^ Firestone 8000 Tire Order by Ford Ford Chooses
Firestone for Model T Jump up ^ Canada manufactures first Firestone tire Archived 2010 - 05 - 10 at the Wayback Machine . Firestone Canada Incorporated Jump up
^ Chris Nichols ( 2013 - 03 - 01 ) . `` DispL. A. Case # 52 : L.A. 's First Firestone Tire - Los Angeles Magazine '' . Lamag.com . Retrieved 2015 - 12 - 17 .
Jump up ^ Lost Memphis Firestone Tire & Rubber Company Jump up ^ Firestone Closings Jump up ^ Peck , Merton J.
</item 1>

<item 2>
Firestone Tire and Rubber Company
Friends say Ford wanted to be near the ocean and picked Long Beach and suggested Firestone go to South Gate , California . The tiny community southeast of
Downtown was mostly agriculture at the time and Firestone found 40 acres of beanfield to house his new manufacturing plant . Architects Curlett and Beelman
created a spectacular four - storey Italianate complex , with its own power plant and gorgeous polychrome murals by Gladding McBean depicting the tire and
rubber - making process . A year after the plant opened in 1928 it doubled in size . By 1954 , when they added the Corporal guided missile to their offerings ,
the plant was nearly a million square feet . The town grew around Firestone , they named the main boulevard through town after Harvey , and Los Angeles became
the number one tire market in the country . By the mid-1970s Ford and GM had massive layoffs as Firestone and other manufacturers opened new plants in non-union
locales like Wilson , North Carolina . After much downsizing the end came in 1980 when 1,300 workers were laid off and the plant closed .
</item 2>

<item 3>
Firestone Tire and Rubber Company
The combined Bridgestone / Firestone North American operations are now based in Nashville , Tennessee . In 2008 , the companies celebrated a 20 - year
anniversary of the merger , and changed the tire division name to Bridgestone Americas Tire Operations , LLC . In 2012 , Bridgestone Americas opened a $100
million technical center in Akron . Firestone tires used for the Indianapolis 500 race . Apart from tires , several companies and divisions operate with the
Firestone brand in its name . These companies include Firestone Building Products , LLC , Firestone Industrial Products , LLC , Firestone Complete Auto Care ,
Firestone Natural Rubber Company and Firestone Specialty Products . Firestone Building Products and Firestone Industrial Products are headquartered in
Indianapolis , Indiana and operate in 23 U.S. states and ten countries , with 11,000 employees worldwide . The company reports annual sales of more than $2.5
billion . In 1972 , Firestone received a ten - year import `` concession '' by the Kenyan government to secure Firestone 's investment in a domestic tire plant
, which gave it a virtual monopoly . This included both general price and foreign exchange controls .
</item 3>

<item 4>
Firestone Tire and Rubber Company
East Los Angeles College has proposed a new satellite campus at the site . In 1928 the company built a factory in Brentford , England , a longtime Art Deco
landmark on a major route into the city . This closed in 1979 . In 1936 the company opened a plant in Memphis , Tennessee . With a work force exceeding 3,000
employees , the Memphis plant was the largest tire manufacturer in the company 's worldwide operation . On July 1 , 1963 , the company celebrated the production
of 100 million tires in Memphis . The plant was closed in 1982 . During World War II the company was called on by the U.S. Government to make artillery shells ,
aluminum kegs for food transport and rubberized military products . Firestone ranked 55th among United States corporations in the value of World War II military
production contracts . In the 1940s , Firestone was given a defense contract to produce plastic helmet liners . While outproduced by Westinghouse Electric they
still made a fair amount for the M1 Helmet .
</item 4>

<item 5>
Firestone Tire and Rubber Company
Founded on August 3 , 1900 , the company initiated operations with 12 employees . Together , Firestone and Goodyear were the largest suppliers of automotive
tires in North America for over 75 years . In 1906 Henry Ford chose Firestone for Model T original equipment tires . In 1918 , Firestone Tire and Rubber Company
of Canada was incorporated in Hamilton , Ontario and in 1922 , the first Canadian - made tire rolled off the line on September 15 . During the 1920s , Firestone
produced the Oldfield tire , named for racing driver Barney Oldfield . In 1926 , the company opened one of the world 's biggest rubber plantations in Liberia ,
West Africa , spanning more than 1 million acres . 1926 was also the year that the company opened its first Firestone Complete Auto Care store ( Firestone
Complete Auto Care is a division of Firestone that offers automotive maintenance and repair ) . In 1927 , Henry Ford and tire maven Harvey Firestone took a trip
to Los Angeles to select locations for their new factories .
</item 5>

<item 6>
Firestone Tire and Rubber Company
External links ( edit ) Wikimedia Commons has media related to Firestone Tire and Rubber Company . Official website Century of Progress from the Summit Memory
Project Bridgestone Corporation Subsidiaries Firestone Diversified Products Firestone Tire and Rubber Company Firestone - Apsley Rubber Company Products
Bridgestone Picnica Helmz Facilities Firestone Tyre Factory People Harvey Samuel Firestone Related Akron Firestone Non-Skids Bridgestone Arena Bridgestone
Museum of Art Bridgestone Open Firestone and Ford tire controversy Firestone Country Club International Labor Rights Forum v Firestone Tire and Rubber Co The
Voice of Firestone WGC - Bridgestone Invitational Automotive industry in the United States Automotive industry Economy of the United States Transportation in
the United States American vehicle manufacturers ( list ) AGCO Challenger Tractor Massey Ferguson AM General American Expedition Vehicles American Growler Amp
Electric Vehicles Anteros Coachworks Arcimoto Armour Group , Inc .
</item 6>

<item 7>
Firestone Tire and Rubber Company
Willys Marathon Motor Works Marmon Roosevelt Marvel Motors Matbro Mercer Monaco Coach Muntz Car Company Oliver Farm Equipment Packard Peerless Motor Company
Pierce - Arrow Sebring Vanguard Sterling Trucks Studebaker Erskine Rockne Stutz Twentieth Century Motor Car Corporation United Defense VL White Wildfire Defunct
factories Brampton Assembly ( AMC ) Diamond - Star Motors Fiat Chrysler factories closed Ford factories closed General Motors factories closed NUMMI Packard
Automotive Plant Volkswagen Westmoreland Assembly Related topics AAA Chicago Auto Show Interstate Highway System National Highway Traffic Safety Administration
New York International Auto Show North American International Auto Show SAE International ( 1 ) Although New Flyer is Canadian , their Subsidiaries , NABI and
Motor Coach Industries , are headquartered in the U.S. ( 2 ) Former meaning the company is no longer in the automotive manufacturing business Portal Retrieved
from `` https://en.wikipedia.org/w/index.php?title=Firestone_Tire_and_Rubber_Company&oldid=806664689 '' Categories : 1900 establishments in Ohio American
companies established in 1900 Tire manufacturers Automotive companies of the United States Companies based in Akron , Ohio Manufacturing companies based in
Nashville , Tennessee Bridgestone U.S.
</item 7>

<item 8>
Firestone Tire and Rubber Company
Firestone tire and Rubber company - wikipedia Firestone tire and Rubber company Jump to : navigation , search Firestone Tire and Rubber Company Type Subsidiary
Industry Manufacturing Founded Akron , Ohio , United States ( August 3 , 1900 ; 117 years ago ( 1900 - 08 - 03 ) ) Founder Harvey Firestone Headquarters 200 4th
Ave S , Nashville , Tennessee , United States Area served Worldwide Key people Gordon Knapp ( CEO and President ) Bill Thompson ( COO ) Products Tires Revenue
US $ 2.09 billion ( 2004 ) Number of employees 40,000 Parent Bridgestone ( 1988 -- present ) Website firestone.com The Firestone Tire and Rubber Company is an
American tire company founded by Harvey Samuel Firestone in 1900 to supply pneumatic tires for wagons , buggies , and other forms of wheeled transportation
common in the era . Firestone soon saw the huge potential for marketing tires for automobiles . The company was a pioneer in the mass production of tires .
Harvey Firestone had a friendship with Henry Ford .
</item 8>

<item 9>
Firestone Tire and Rubber Company
Retrieved 2007 - 03 - 21 . Jump up ^ Isidore , Chris ( May 21 , 2001 ) . `` Firestone severs ties to Ford '' . CNN . Jump up ^ NHTSA Recall Campaign 01T - 016
Q4 Vehicle Safety Recall Quarterly Report Information , January 25 , 2008 ( Amended March 26 , 2009 ) ( from the National Highway Traffic Safety Administration
) ^ Jump up to : Channel 2 Investigation Finds Recalled Tires for Sale , November 4 , 2013 Archived November 5 , 2013 , at the Wayback Machine . Jump up ^ ``
Firestone Jingle '' ( QuickTime ) . Firestone . Retrieved 2007 - 03 - 22 . External links ( edit ) Wikimedia Commons has media related to Firestone Tire and
Rubber Company .
</item 9>

<item 10>
Firestone Tire and Rubber Company
Harvey Firestone had a friendship with Henry Ford . Firestone used this relationship to become the original equipment supplier of Ford Motor Company automobiles
, and was also active in the replacement market . In 1988 , the company was sold to the Japanese Bridgestone Corporation . Contents ( hide ) 1 History 1.1 Early
- to - mid-20th century 1.2 Restructuring and sale to Bridgestone 2 In motorsport 3 Corporate troubles 3.1 Great American streetcar scandal 3.2 Firestone 500
tread separation problem 3.3 Liberian rubber plantation 3.4 Ford Explorer rollover problem 4 Television advertisement jingle 5 Other brands 6 See also 7
References 8 External links History ( edit ) Early - to - mid-20th century ( edit ) The First Firestone store Firestone newspaper ad from 1919 , explaining the
benefits of their cord tire . Firestone was originally based in Akron , Ohio , also the hometown of its archrival , Goodyear Tire and Rubber Company , and
another two mid-sized competitors , General Tire and Rubber and BF Goodrich .
</item 10>
=== END ITEM id=f9d3109b1e530d42 ===

=== ITEM id=9839848723c1d21a ===
Gold sentences and facts:
Sentence 0: Central London Property Trust Ltd v High Trees House Ltd ( 1947 ) KB 130 ( or the High Trees case ) is an English contract law decision in the High Court . It reaffirmed and extended the doctrine of promissory estoppel in contract law in England and Wales . However , the most significant part of the judgment is obiter dicta as it relates to hypothetical facts ; that is , the landlord did not seek repayment of the full wartime rent .
  Fact 0.0: Central London Property Trust Ltd v High Trees House Ltd is cited as (1947) KB 130.
  Fact 0.1: The case Central London Property Trust Ltd v High Trees House Ltd is also called the High Trees case.
  Fact 0.2: The High Trees case is an English contract law decision in the High Court.
  Fact 0.3: The High Trees case reaffirmed the doctrine of promissory estoppel in contract law in England and Wales.
  Fact 0.4: The High Trees case extended the doctrine of promissory estoppel in contract law in England and Wales.
  Fact 0.5: The most significant part of the High Trees judgment is obiter dicta.
  Fact 0.6: The obiter dicta in the High Trees judgment relates to hypothetical facts.
  Fact 0.7: In the hypothetical facts, the landlord did not seek repayment of the full wartime rent.

Retrieved items:
<item 1>
Central London Property Trust Ltd v High Trees House Ltd
It was a representation , in effect , as to the future , namely , that payment of the rent would not be enforced at the full rate but only at the reduced rate .
Such a representation would not give rise to an estoppel , because , as was said in Jorden v. Money ( 1854 ) 5 H.L.C. 185 , a representation as to the future
must be embodied as a contract or be nothing . But what is the position in view of developments in the law in recent years ? The law has not been standing still
since Jorden v. Money ( 1854 ) 5 H.L.C. 185 . There has been a series of decisions over the last fifty years which , although they are said to be cases of
estoppel are not really such . They are cases in which a promise was made which was intended to create legal relations and which , to the knowledge of the
person making the promise , was going to be acted on by the person to whom it was made and which was in fact so acted on . In such cases the courts have said
that the promise must be honoured .
</item 1>

<item 2>
Central London Property Trust Ltd v High Trees House Ltd
Here it was binding as covering the period down to the early part of 1945 , and as from that time full rent is payable . I therefore give judgment for the
plaintiff company for the amount claimed . Significance ( edit ) Advances have been made in promissory estoppel since its inception in High Trees to create a
new inroad into the rule in Pinnel 's case that an agreement to accept part payment of a debt in full satisfaction of it is unenforceable for want of
consideration . Denning commented that such an agreement should now be enforceable under the doctrine of promissory estoppel , and indeed the plaintiff did not
seek the full debt on the basis of what was fair and , perhaps , thought was the law . However , the courts were at first reluctant to overrule or distinguish
cases like Pinnel 's case and Foakes v Beer having formed part of the common law for so long . Lady Justice Arden in Collier v P & MJ Wright ( Holdings ) Ltd (
2007 ) accepted in principle that High Trees could be used to extinguish a creditor 's right to full payment of a debt in such circumstances .
</item 2>

<item 3>
Central London Property Trust Ltd v High Trees House Ltd
In Amalgamated Investment Co v Texas Bank it was decided that proprietory estoppel can act as a sword and not merely as a shield ( that is , it can be used as a
cause of action rather than merely providing a defence to an action ) . See also ( edit ) English contract law Estoppel Promise References ( edit ) Jump up ^
Collier v P & MJ Wright ( Holdings ) Ltd ( 2007 ) EWCA Civ 1329 Jump up ^ Amalgamated Investment Co v Texas Bank , ( 1982 ) EWHC 84 ( QB ) .
</item 3>

<item 4>
Graph query row:
organization: Central London Property Trust Ltd
organization_relationship: PARTICIPATED_IN
other_organization: High Trees House Ltd
other_organization_relationship: PARTICIPATED_IN
person: Denning J
person_relationship: PARTICIPATED_IN
event: Central London Property Trust Ltd v High Trees House Ltd
concept: promissory estoppel
concept_relationship: RELATED_TO
</item 4>

<item 5>
Graph query row:
organization: Central London Property Trust Ltd
organization_relationship: RELATED_TO
other_organization: High Trees House Ltd
other_organization_relationship: RELATED_TO
person: Denning J
person_relationship: PARTICIPATED_IN
event: Central London Property Trust Ltd v High Trees House Ltd
concept: promissory estoppel
concept_relationship: RELATED_TO
</item 5>

<item 6>
Central London Property Trust Ltd v High Trees House Ltd
for last two quarters of 1945 ) . Judgment ( edit ) Based on previous judgments as Hughes v Metropolitan Railway Co , Denning J held that the full rent was
payable from the time that the flats became fully occupied in mid-1945 . However , he continued in an obiter statement that if Central London had tried to claim
for the full rent from 1940 onwards , they would not have been able to . This was reasoned on the basis that if a party leads another party to believe that he
will not enforce his strict legal rights , then the Courts will prevent him from doing so at a later stage . Being obiter dicta and in a court of first instance
this was doubly not a binding precedent , yet it essentially created the doctrine of promissory estoppel .
</item 6>

<item 7>
Central London Property Trust Ltd v High Trees House Ltd
Retrieved from `` https://en.wikipedia.org/w/index.php?title=Central_London_Property_Trust_Ltd_v_High_Trees_House_Ltd&oldid=837737647 '' Categories : English
contract case law Lord Denning cases English enforceability case law English estoppel case law 1946 in British law High Court of Justice cases 1946 in case law
Hidden categories : Use British English from August 2015 Use dmy dates from February 2012 Talk Contents About Wikipedia Add links This page was last edited on
22 April 2018 , at 18 : 40 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site ,
you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About
Wikipedia
</item 7>

<item 8>
Central London Property Trust Ltd v High Trees House Ltd
In such cases the courts have said that the promise must be honoured . The cases to which I particularly desire to refer are : Fenner v. Blake ( 1900 ) 1 Q.B.
426 , In re Wickham ( 1917 ) 34 T.L.R. 158 , Re William Porter & Co. , Ld. ( 1937 ) 2 All E.R. 361 and Buttery v. Pickard ( 1946 ) W.N. 25 . As I have said they
are not cases of estoppel in the strict sense . They are really promises - promises intended to be binding , intended to be acted on , and in fact acted on .
Jorden v. Money ( 1854 ) 5 H.L.C. 185 can be distinguished , because there the promisor made it clear that she did not intend to be legally bound , whereas in
the cases to which I refer the proper inference was that the promisor did intend to be bound .
</item 8>

<item 9>
Graph query row:
organization: Central London Property Trust Ltd
organization_relationship: PARTICIPATED_IN
other_organization: High Trees House Ltd
other_organization_relationship: PARTICIPATED_IN
person: Lord Denning
person_relationship: RELATED_TO
event: Central London Property Trust Ltd v High Trees House Ltd
concept: promissory estoppel
concept_relationship: RELATED_TO
</item 9>

<item 10>
Central London Property Trust Ltd v High Trees House Ltd
In each case the court held the promise to be binding on the party making it , even though under the old common law it might be difficult to find any
consideration for it . The courts have not gone so far as to give a cause of action in damages for the breach of such a promise , but they have refused to allow
the party making it to act inconsistently with it . It is in that sense , and that sense only , that such a promise gives rise to an estoppel . The decisions
are a natural result of the fusion of law and equity : for the cases of Hughes v. Metropolitan Ry. Co. ( 1877 ) 2 App . Cas. 439 , 448 , Birmingham and District
Land Co. v. London & North Western Ry. Co. ( 1888 ) 40 Ch . D. 268 , 286 and Salisbury ( Marquess ) v. Gilmore ( 1942 ) 2 K.B. 38 , 51 , afford a sufficient
basis for saying that a party would not be allowed in equity to go back on such a promise .
</item 10>

<item 11>
Graph query row:
organization: Central London Property Trust Ltd
organization_relationship: PARTICIPATED_IN
other_organization: High Trees House Ltd
other_organization_relationship: RELATED_TO
person: Denning J
person_relationship: PARTICIPATED_IN
event: Central London Property Trust Ltd v High Trees House Ltd
concept: promissory estoppel
concept_relationship: RELATED_TO
</item 11>

<item 12>
Central London Property Trust Ltd v High Trees House Ltd
Central London Property Trust Ltd v. High Trees House Ltd - wikipedia Central London Property Trust Ltd v. High Trees House Ltd Jump to : navigation , search
High Trees case Court King 's Bench Division Decided 18 July 1946 Citation ( s ) ( 1947 ) KB 130 ; ( 1956 ) 1 All ER 256 ; 62 TLR 557 ; ( 1947 ) LJR 77 ; 175 LT
333 Court membership Judge ( s ) sitting Denning J Keywords Promissory estoppel Central London Property Trust Ltd v High Trees House Ltd ( 1947 ) KB 130 ( or
the High Trees case ) is an English contract law decision in the High Court . It reaffirmed and extended the doctrine of promissory estoppel in contract law in
England and Wales . However , the most significant part of the judgment is obiter dicta as it relates to hypothetical facts ; that is , the landlord did not
seek repayment of the full wartime rent .
</item 12>

<item 13>
Graph query row:
organization: Central London Property Trust Ltd
organization_relationship: RELATED_TO
other_organization: High Trees House Ltd
other_organization_relationship: PARTICIPATED_IN
person: Denning J
person_relationship: PARTICIPATED_IN
event: Central London Property Trust Ltd v High Trees House Ltd
concept: promissory estoppel
concept_relationship: RELATED_TO
</item 13>

<item 14>
Central London Property Trust Ltd v High Trees House Ltd
Denning J held estoppel to be applicable if ... ... a promise was made which was intended to create legal relations and which , to the knowledge of the person
making the promise , was going to be acted on by the person to whom it was made and which was in fact so acted on . Contents ( hide ) 1 Facts 2 Judgment 3
Significance 4 See also 5 References Facts ( edit ) In 1937 , High Trees House Ltd leased a block of flats in Clapham , London , for a rate £ 2500 / year from
Central London Property Trust Ltd . Due to the conditions during the beginning of World War II occupancy rates were drastically lower than normal . In January
1940 , to ameliorate the situation , the parties made an agreement in writing to reduce rent by half . However , neither party stipulated the period for which
this reduced rental was to apply . Over the next five years , High Trees paid the reduced rate while the flats began to fill , and by 1945 , the flats were back
at full occupancy . Central London sued for payment of the full rental costs from June 1945 onwards ( i.e.
</item 14>

<item 15>
Central London Property Trust Ltd v High Trees House Ltd
I am satisfied that the promise was understood by all parties only to apply under the conditions prevailing at the time when it was made , namely , when the
flats were only partially let , and that it did not extend any further than that . When the flats became fully let , early in 1945 , the reduction ceased to
apply . In those circumstances , under the law as I hold it , it seems to me that rent is payable at the full rate for the quarters ending September 29 and
December 25 , 1945 . If the case had been one of estoppel , it might be said that in any event the estoppel would cease when the conditions to which the
representation applied came to an end , or it also might be said that it would only come to an end on notice . In either case it is only a way of ascertaining
what is the scope of the representation . I prefer to apply the principle that a promise intended to be binding , intended to be acted on and in fact acted on ,
is binding so far as its terms properly apply .
</item 15>
=== END ITEM id=9839848723c1d21a ===

=== ITEM id=1c571b55e74ec0bf ===
Gold sentences and facts:
Sentence 0: The village of New Earswick was built as a genuine mixed community . There was housing for both workers and managers , in a green setting with gardens for each home with its own 2 fruit trees . It was founded by the York philanthropist , Joseph Rowntree , who was quoted as saying , `` I do not want to establish communities bearing the stamp of charity but rather of rightly ordered and self governing communities '' . The first 28 houses were built between 1902 and 1904 by the architect Raymond Unwin , after which the Joseph Rowntree Village Trust ( now the Joseph Rowntree Housing Trust ) was established to continue building and manage the new village . The Joseph Rowntree Housing Trust is part of the Joseph Rowntree Foundation .
  Fact 0.0: The village of New Earswick was built as a genuine mixed community.
  Fact 0.1: New Earswick had housing for both workers and managers.
  Fact 0.2: New Earswick was set in a green setting.
  Fact 0.3: New Earswick had gardens for each home.
  Fact 0.4: Each home in New Earswick had its own 2 fruit trees.
  Fact 0.5: New Earswick was founded by the York philanthropist Joseph Rowntree.
  Fact 0.6: Joseph Rowntree said: 'I do not want to establish communities bearing the stamp of charity but rather of rightly ordered and self governing communities.'
  Fact 0.7: The first 28 houses of New Earswick were built between 1902 and 1904.
  Fact 0.8: The first 28 houses of New Earswick were built by the architect Raymond Unwin.
  Fact 0.9: The Joseph Rowntree Village Trust was established after the first houses were built.
  Fact 0.10: The Joseph Rowntree Village Trust was established to continue building and manage the new village.
  Fact 0.11: The Joseph Rowntree Village Trust is now called the Joseph Rowntree Housing Trust.
  Fact 0.12: The Joseph Rowntree Housing Trust is part of the Joseph Rowntree Foundation.
Sentence 1: As a result of the report , Joseph Rowntree 's conviction that it must be possible to provide better housing for people on low incomes led him to acquire 150 acres of land near the village of Earswick , two and a half miles to the north of the centre of York . The planner Raymond Unwin and the architect Barry Parker were commissioned to produce an overall plan for a new ' garden ' village and the detailed designs for its first houses . They also designed the garden cities of Letchworth and Welwyn Garden City .
  Fact 1.0: As a result of the report, Joseph Rowntree acquired 150 acres of land near the village of Earswick.
  Fact 1.1: Joseph Rowntree was convinced that it must be possible to provide better housing for people on low incomes.
  Fact 1.2: The 150 acres of land are two and a half miles north of the centre of York.
  Fact 1.3: The planner Raymond Unwin and the architect Barry Parker were commissioned to produce an overall plan for a new 'garden' village.
  Fact 1.4: Raymond Unwin and Barry Parker were commissioned to produce the detailed designs for the first houses of the new 'garden' village.
  Fact 1.5: Raymond Unwin also designed the garden city of Letchworth.
  Fact 1.6: Barry Parker also designed the garden city of Letchworth.
  Fact 1.7: Raymond Unwin also designed the garden city of Welwyn Garden City.
  Fact 1.8: Barry Parker also designed the garden city of Welwyn Garden City.

Retrieved items:
<item 1>
New Earswick
Jump up ^ `` New Earswick All Blacks '' . Archived from the original on 16 January 2008 . Jump up ^ Havard , Jane ( 1992 ) . A Bit On The Side . Menoshire . p.
1 . ISBN 0947981667 . External links ( edit ) `` New Earswick , York '' . The Joseph Rowntree Housing Trust . Retrieved 4 March 2017 .
</item 1>

<item 2>
New Earswick
There never has been , although the Sports Club obtained a drink 's license in the 1980s . Governance ( edit ) New Earswick is part of the Huntington & New
Earswick Ward of the City of York Council . As of 2010 it is represented by Keith Hyman , Keith Orrel and Carol Runciman who are all members of the local
Liberal Democrat Party . A Parish Council was created in 1934 consisting of ten elected councillors and a clerk . Community ( edit ) There is a local community
centre , the Folk Hall , which hosts activities such as keep - fit , yoga , snooker , and a junior youth club . New Earswick has recreational facilities ,
including a swimming pool ( due for closure without public consultation ) , tennis courts and football , rugby and cricket pitches at the New Earswick Sports
and Social Club . There were two open access play areas close to the Folk Hall , including a hard surface play area which have recently been bulldozed for a
development of houses .
</item 2>

<item 3>
New Earswick
The planner Raymond Unwin and the architect Barry Parker were commissioned to produce an overall plan for a new ' garden ' village and the detailed designs for
its first houses . They also designed the garden cities of Letchworth and Welwyn Garden City . The building of New Earswick created a balanced village community
where rents were kept low , but still represented a modest commercial return on the capital invested . Houses were open to any working people , not just
Rowntree employees . The village was to be a demonstration of good practice . The Trust Deed of the Joseph Rowntree Village Trust , which was set up in 1904 to
build and manage New Earswick , safeguarded generous open green space . All the grass verges were planted with trees after which almost all the roads are named
. The village was built with bricks that were made in the brickworks on the outskirts of New Earswick . From 1950 the brickyard , which closed down in the 1930s
, was developed into a nature reserve . Due to their religious beliefs , The Rowntree family decreed there would not be a public house in the village .
</item 3>

<item 4>
New Earswick
The first 28 houses were built between 1902 and 1904 by the architect Raymond Unwin , after which the Joseph Rowntree Village Trust ( now the Joseph Rowntree
Housing Trust ) was established to continue building and manage the new village . The Joseph Rowntree Housing Trust is part of the Joseph Rowntree Foundation .
The village had contemporaries such as Bournville , Saltaire , Port Sunlight and others . This was in sharp contrast to the slums that had developed in York and
other cities during the previous century , the deprivation of which had been revealed in Seebohm Rowntree 's report of 1901 , entitled Poverty : a study of town
life . It revealed appalling statistics of dark , overcrowded and insanitary housing . As a result of the report , Joseph Rowntree 's conviction that it must be
possible to provide better housing for people on low incomes led him to acquire 150 acres of land near the village of Earswick , two and a half miles to the
north of the centre of York .
</item 4>

<item 5>
New Earswick
The Joseph Rowntree Housing Trust . Retrieved 4 March 2017 . Retrieved from `` https://en.wikipedia.org/w/index.php?title=New_Earswick&oldid=826057861 ''
Categories : Model villages Villages and areas in the City of York Civil parishes in North Yorkshire Rowntree family Hidden categories : All articles with dead
external links Articles with dead external links from November 2017 Articles with permanently dead external links Use British English from October 2013 Use dmy
dates from October 2013 Articles with OS grid coordinates Coordinates on Wikidata All articles with unsourced statements Articles with unsourced statements from
April 2013 Talk Contents About Wikipedia Cebuano Gaeilge Nederlands Polski Svenska Edit links This page was last edited on 16 February 2018 , at 23 : 16 . Text
is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and
Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 5>

<item 6>
New Earswick
There is a library based in the Integrated Children 's Centre at the local primary school , a doctor 's surgery and a range of shops , including two general
stores , a pet shop and Post Office . As of 2017 the Post Office is situated within the Folk Hall . The village has 36 allotments in two areas , Willow Bank and
Sleeper Path . In the 1960s , the Folk Hall was a live music venue featuring the likes of Procol Harum and Pink Floyd . Geography ( edit ) The River Foss runs
along the eastern boundary of New Earswick with the York to Scarborough Line running to the west . The Nestle factory and grounds mark the southern border with
the northern border being the A1237 York Outer Ring Road . The village was built using bricks from the brickworks on the outskirts of the village . From 1950
the brickyard was developed into a nature reserve . The reserve is the home of New Earswick Angling Club , which was formed in 1948 .
</item 6>

<item 7>
New Earswick
New Earswick - wikipedia New Earswick Jump to : navigation , search New Earswick New Earswick New Earswick shown within North Yorkshire Population 2,737 ( 2011
) OS grid reference SE609555 Civil parish New Earswick Unitary authority City of York Ceremonial county North Yorkshire Region Yorkshire and the Humber Country
England Sovereign state United Kingdom Post town YORK Postcode district YO32 Dialling code 01904 Police North Yorkshire Fire North Yorkshire Ambulance Yorkshire
EU Parliament Yorkshire and the Humber UK Parliament York Outer List of places UK England Yorkshire 53 ° 59 ′ 31 '' N 1 ° 04 ′ 21 '' W ﻿ / ﻿ 53.992020 ° N
1.072380 ° W ﻿ / 53.992020 ; - 1.072380 Coordinates : 53 ° 59 ′ 31 '' N 1 ° 04 ′ 21 '' W ﻿ / ﻿ 53.992020 ° N 1.072380 ° W ﻿ / 53.992020 ; - 1.
</item 7>

<item 8>
New Earswick
Steve Webster , 10 times FIM Sidecar World Championships , world champion was born and raised in the village. . Gallery ( edit ) Views of New Earswick The Folk
Hall in New Earswick White Rose Avenue in New Earswick Westfield Beck behind housing on the west of New Earswick References ( edit ) Jump up ^ `` Civil Parish
population 2011 '' . Neighbourhood Statistics . Office for National Statistics . Retrieved 19 February 2016 . Jump up ^ `` Village History '' . Retrieved 12
December 2010 . Jump up ^ `` Village planning '' . Retrieved 12 December 2010 . Jump up ^ `` New Earswick : a village of new ideas '' . BBC . Retrieved 29 March
2014 . Jump up ^ `` Couoncillors '' . Retrieved 12 December 2010 . ^ Jump up to : `` Village Information '' ( PDF ) . Archived from the original ( PDF ) on 4
October 2013 . Retrieved 12 December 2010 .
</item 8>

<item 9>
New Earswick
Economy ( edit ) Unlike the villages of Port Sunlight , Saltaire and Bournville the village was not designed specifically for the workforce of the nearby Joseph
Rowntree Cocoa Works although some do earn their living at the factory which was taken over by Swiss firm , Nestle . Transport ( edit ) Transdev York operate a
bus service that stops in the village as part of the Askham Bar to Clifton Moor route . First Group operate two bus services that stop in the village as part of
the Wigginton to Acomb and Haxby to York City Centre routes . New Earswick was served by Earswick railway station on the York to Beverley Line between 1847 and
1965 . Education ( edit ) There are both primary and secondary schools in New Earswick . New Earswick Primary School was built in 1912 . The primary school is
one of eight Integrated Children 's Centres in York . Joseph Rowntree School is a secondary school and was built in 1941 . In 2006 , the Joseph Rowntree School
secured £ 27million for a complete rebuild that was completed in late 2009 .
</item 9>

<item 10>
New Earswick
072380 ° W ﻿ / 53.992020 ; - 1.072380 New Earswick is a model village and civil parish in the unitary authority of City of York in North Yorkshire , England ,
near the River Foss , north of York and south of Haxby . According to the 2001 census the parish had a population of 2,812 , reducing to 2,737 at the 2011
Census . Before 1996 it had been part of the Ryedale district . Contents ( hide ) 1 History 2 Governance 3 Community 4 Geography 5 Economy 6 Transport 7
Education 8 Religious sites 9 Sport 10 Gallery 11 References 12 External links History ( edit ) The village of New Earswick was built as a genuine mixed
community . There was housing for both workers and managers , in a green setting with gardens for each home with its own 2 fruit trees . It was founded by the
York philanthropist , Joseph Rowntree , who was quoted as saying , `` I do not want to establish communities bearing the stamp of charity but rather of rightly
ordered and self governing communities '' .
</item 10>
=== END ITEM id=1c571b55e74ec0bf ===

=== ITEM id=68b1b54a64f42944 ===
Gold sentences and facts:
Sentence 0: `` Baby , Now That I 've Found You '' ( Tony Macaulay , John MacLeod ) -- 3 : 49 ( Previously Unreleased ) `` Oh , Atlanta '' ( Mick Ralphs ) -- 4 : 41 ( Previously Unreleased ) `` Broadway '' ( Sidney & Suzanne Cox ) -- 3 : 57 ( Previously Unreleased ) `` Every Time You Say Goodbye '' ( John Pennell ) -- 3 : 13 ( From the Album `` Every Time You Say Goodbye '' by Alison Krauss & Union Station ) `` Tonight I 'll Be Lonely Too '' ( Sidney Cox ) -- 3 : 25 ( From the Album `` I 've Got That Old Feeling '' by Alison Krauss ) `` Teardrops Will Kiss the Morning Dew '' ( Paul Craft ) -- 3 : 38 ( From the Album `` Two Highways '' by Alison Krauss & Union Station ) `` Sleep On '' ( Nelson Mandrell ) -- 2 : 23 ( From the Album `` Too Late to Cry '' by Alison Krauss ) `` When God Dips His Pen of Love in My Heart '' ( Cleavant Derricks ) -- 2 : 26 ( From the Album `` Everybody 's Reaching Out for Someone '' by The Cox Family ) `` I Will '' ( John Lennon , Paul McCartney ) -- 4 : 04 ( From the Album `` Within Reach '' by Tony Furtado ) `` I Do n't Believe You 've Met My Baby '' ( Autry Inman ) -- 3 : 10 ( From the Album `` Slide Rule '' by Jerry Douglas ) `` In the Palm of Your Hand '' ( Ron Block ) -- 3 : 24 ( From the Album `` I Know Who Holds Tomorrow '' by Alison Krauss & The Cox Family ) `` When You Say Nothing at All '' ( Paul Overstreet , Don Schlitz ) -- 4 : 21 ( From the Album `` Keith Whitley : A Tribute Album '' performed by Alison Krauss & Union Station )
  Fact 0.0: 'Baby, Now That I've Found You' was written by Tony Macaulay and John MacLeod.
  Fact 0.1: 'Baby, Now That I've Found You' runs 3:49.
  Fact 0.2: 'Baby, Now That I've Found You' is previously unreleased.
  Fact 0.3: 'Oh, Atlanta' was written by Mick Ralphs.
  Fact 0.4: 'Oh, Atlanta' runs 4:41.
  Fact 0.5: 'Oh, Atlanta' is previously unreleased.
  Fact 0.6: 'Broadway' was written by Sidney & Suzanne Cox.
  Fact 0.7: 'Broadway' runs 3:57.
  Fact 0.8: 'Broadway' is previously unreleased.
  Fact 0.9: 'Every Time You Say Goodbye' was written by John Pennell.
  Fact 0.10: 'Every Time You Say Goodbye' runs 3:13.
  Fact 0.11: 'Every Time You Say Goodbye' is from the album 'Every Time You Say Goodbye' by Alison Krauss & Union Station.
  Fact 0.12: 'Tonight I'll Be Lonely Too' was written by Sidney Cox.
  Fact 0.13: 'Tonight I'll Be Lonely Too' runs 3:25.
  Fact 0.14: 'Tonight I'll Be Lonely Too' is from the album 'I've Got That Old Feeling' by Alison Krauss.
  Fact 0.15: 'Teardrops Will Kiss the Morning Dew' was written by Paul Craft.
  Fact 0.16: 'Teardrops Will Kiss the Morning Dew' runs 3:38.
  Fact 0.17: 'Teardrops Will Kiss the Morning Dew' is from the album 'Two Highways' by Alison Krauss & Union Station.
  Fact 0.18: 'Sleep On' was written by Nelson Mandrell.
  Fact 0.19: 'Sleep On' runs 2:23.
  Fact 0.20: 'Sleep On' is from the album 'Too Late to Cry' by Alison Krauss.
  Fact 0.21: 'When God Dips His Pen of Love in My Heart' was written by Cleavant Derricks.
  Fact 0.22: 'When God Dips His Pen of Love in My Heart' runs 2:26.
  Fact 0.23: 'When God Dips His Pen of Love in My Heart' is from the album 'Everybody's Reaching Out for Someone' by The Cox Family.
  Fact 0.24: 'I Will' was written by John Lennon and Paul McCartney.
  Fact 0.25: 'I Will' runs 4:04.
  Fact 0.26: 'I Will' is from the album 'Within Reach' by Tony Furtado.
  Fact 0.27: 'I Don't Believe You've Met My Baby' was written by Autry Inman.
  Fact 0.28: 'I Don't Believe You've Met My Baby' runs 3:10.
  Fact 0.29: 'I Don't Believe You've Met My Baby' is from the album 'Slide Rule' by Jerry Douglas.
  Fact 0.30: 'In the Palm of Your Hand' was written by Ron Block.
  Fact 0.31: 'In the Palm of Your Hand' runs 3:24.
  Fact 0.32: 'In the Palm of Your Hand' is from the album 'I Know Who Holds Tomorrow' by Alison Krauss & The Cox Family.
  Fact 0.33: 'When You Say Nothing at All' was written by Paul Overstreet and Don Schlitz.
  Fact 0.34: 'When You Say Nothing at All' runs 4:21.
  Fact 0.35: 'When You Say Nothing at All' is from the album 'Keith Whitley: A Tribute Album' performed by Alison Krauss & Union Station.

Retrieved items:
<item 1>
Now That I've Found You: A Collection
Atlanta '' ( Mick Ralphs ) -- 4 : 41 ( Previously Unreleased ) `` Broadway '' ( Sidney & Suzanne Cox ) -- 3 : 57 ( Previously Unreleased ) `` Every Time You Say
Goodbye '' ( John Pennell ) -- 3 : 13 ( From the Album `` Every Time You Say Goodbye '' by Alison Krauss & Union Station ) `` Tonight I 'll Be Lonely Too '' (
Sidney Cox ) -- 3 : 25 ( From the Album `` I 've Got That Old Feeling '' by Alison Krauss ) `` Teardrops Will Kiss the Morning Dew '' ( Paul Craft ) -- 3 : 38 (
From the Album `` Two Highways '' by Alison Krauss & Union Station ) `` Sleep On '' ( Nelson Mandrell ) -- 2 : 23 ( From the Album `` Too Late to Cry '' by
Alison Krauss ) `` When God Dips His Pen of Love in My Heart '' ( Cleavant Derricks ) -- 2 : 26 ( From the Album `` Everybody 's Reaching Out for Someone '' by
The Cox Family ) `` I Will '' ( John Lennon ,
</item 1>

<item 2>
Graph query row:
album: Now That I've Found You: A Collection
release_date: February 7, 1995
kind: album
artist: Alison Krauss
relationship: RELATED_TO
related_type: ['__KGBuilder__', '__Entity__', 'Organization']
related_work_or_entity: Rounder
</item 2>

<item 3>
Now That I've Found You: A Collection
It is a retrospective of the early part of Krauss ' recording career . It includes songs that appeared on her solo albums , albums by Alison Krauss & Union
Station , and some that appeared on an album by Alison Krauss & the Cox Family . It also contains some new material . The album features the hit country single
`` When You Say Nothing at All '' , sold over two million copies earning double platinum RIAA certification . The title single `` Baby , Now That I 've Found
You '' won the Grammy Award for Best Female Country Vocal Performance and `` When You Say Nothing at All '' was named Single of the Year by the Country Music
Association . Contents ( hide ) 1 Track listing 2 Personnel 3 Chart performance 3.1 Album 4 References Track listing ( edit ) `` Baby , Now That I 've Found You
'' ( Tony Macaulay , John MacLeod ) -- 3 : 49 ( Previously Unreleased ) `` Oh ,
</item 3>

<item 4>
Now That I've Found You: A Collection
Paul McCartney ) -- 4 : 04 ( From the Album `` Within Reach '' by Tony Furtado ) `` I Do n't Believe You 've Met My Baby '' ( Autry Inman ) -- 3 : 10 ( From the
Album `` Slide Rule '' by Jerry Douglas ) `` In the Palm of Your Hand '' ( Ron Block ) -- 3 : 24 ( From the Album `` I Know Who Holds Tomorrow '' by Alison
Krauss & The Cox Family ) `` When You Say Nothing at All '' ( Paul Overstreet , Don Schlitz ) -- 4 : 21 ( From the Album `` Keith Whitley : A Tribute Album ''
performed by Alison Krauss & Union Station ) Personnel ( edit ) Alison Krauss : Vocal , Fiddle , Viola Ron Block : Vocal , Acoustic & Electric Guitars , Banjo
Russ Barenberg , Mike Marshall , Scott Nygaard , Tim Stafford : Guitars Jerry Douglas : Guitars , Dobro Sam Bush , Brent Truitt : Mandolin Stuart Duncan :
Fiddle , Harmonica ,
</item 4>

<item 5>
Now That I've Found You: A Collection
Now That I 've Found You : a Collection - wikipedia Now That I 've Found You : a Collection Jump to : navigation , search Now That I 've Found You : A
Collection Compilation album by Alison Krauss Released February 7 , 1995 Genre Bluegrass / Country Length 42 : 31 Label Rounder Producer Alison Krauss & Union
Station Alison Krauss chronology I Know Who Holds Tomorrow ( 1994 ) I Know Who Holds Tomorrow 1994 Now That I 've Found You : A Collection ( 1995 ) So Long So
Wrong ( 1997 ) So Long So Wrong 1997 Singles from Now That I 've Found You : A Collection `` Baby , Now That I 've Found You '' Released : 1995 `` Oh , Atlanta
'' Released : 1995 Professional ratings Review scores Source Rating Allmusic Robert Christgau A − Rolling Stone Now That I 've Found You : A Collection is an
album by Alison Krauss , released February 7 , 1995 . It is a retrospective of the early part of Krauss ' recording career .
</item 5>

<item 6>
Graph query row:
album: Now That I've Found You: A Collection
release_date: February 7, 1995
kind: album
artist: Alison Krauss
relationship: RELATED_TO
related_type: ['__KGBuilder__', '__Entity__', 'Concept']
related_work_or_entity: double platinum RIAA certification
</item 6>

<item 7>
Graph query row:
album: Now That I've Found You: A Collection
release_date: February 7, 1995
kind: album
artist: Alison Krauss
relationship: RELATED_TO
related_type: ['__KGBuilder__', '__Entity__', 'Concept']
related_work_or_entity: U.S. Billboard 200
</item 7>

<item 8>
Graph query row:
album: Now That I've Found You: A Collection
release_date: February 7, 1995
kind: album
artist: Alison Krauss
relationship: RELATED_TO
related_type: ['__KGBuilder__', '__Entity__', 'Concept']
related_work_or_entity: Bluegrass
</item 8>

<item 9>
Now That I've Found You: A Collection
Close and Personal Retrieved from `` https://en.wikipedia.org/w/index.php?title=Now_That_I%27ve_Found_You:_A_Collection&oldid=808197945 '' Categories : 1995
compilation albums Alison Krauss & Union Station albums Rounder Records compilation albums Hidden categories : Articles with hAudio microformats Music infoboxes
with deprecated parameters Talk Contents About Wikipedia Add links This page was last edited on 1 November 2017 , at 13 : 34 . About Wikipedia
</item 9>

<item 10>
Graph query row:
album: Now That I've Found You: A Collection
release_date: February 7, 1995
kind: album
artist: Alison Krauss
relationship: RELATED_TO
related_type: ['__KGBuilder__', '__Entity__', 'Concept']
related_work_or_entity: Canadian RPM Country Albums
</item 10>

<item 11>
Now That I've Found You: A Collection
Brent Truitt : Mandolin Stuart Duncan : Fiddle , Harmonica , Cymbals Adam Steffey : Mandolin & Mandola Pete Wasner : Piano Chart performance ( edit ) Album (
edit ) Chart ( 1995 ) Peak position Canadian RPM Country Albums U.S. Billboard 200 13 U.S. Billboard Top Country Albums References ( edit ) Jump up ^ Allmusic
review Jump up ^ Robert Christgau review Jump up ^ Rolling Stone review ^ Jump up to : Alison Krauss Artist Chart Position at Billboard Now That I 've Found You
for Rounder Records . Retrieved August 28 , 2010 . Alison Krauss bio for CMT.com . Retrieved June 7 , 2006 . Now That I 've Found You on Amazon . Retrieved June
7 , 2006 . Preceded by The Woman in Me by Shania Twain RPM Country Albums number - one album October 30 - November 5 ,
</item 11>

<item 12>
Now That I've Found You: A Collection
1995 Succeeded by Starting Over by Reba McEntire ( hide ) Alison Krauss & Union Station Alison Krauss Barry Bales Ron Block Jerry Douglas Dan Tyminski John
Pennell Jeff White Brent Truitt Mike Harman Adam Steffey Tim Stafford Studio albums Different Strokes Too Late to Cry Two Highways I 've Got That Old Feeling
Every Time You Say Goodbye I Know Who Holds Tomorrow So Long So Wrong Forget About It New Favorite Lonely Runs Both Ways Paper Airplane Windy City Compilation
albums Now That I 've Found You : A Collection Live A Hundred Miles or More : A Collection Essential Alison Krauss With Robert Plant Raising Sand Singles ``
When You Say Nothing at All '' `` Baby Now That I 've Found You '' `` Baby Mine '' `` The Lucky One '' `` Missing You '' `` Please Read the Letter '' `` Rich
Woman '' Featured singles `` Teach Your Children '' `` Somewhere in the Vicinity of the Heart '' `` It 's Not Over '' `` Buy Me a Rose '' `` Whiskey Lullaby ''
Related articles Discography Awards and nominations Alison Brown Robert Plant Viktor Krauss Up !
</item 12>

<item 13>
Graph query row:
album: Now That I've Found You: A Collection
release_date: February 7, 1995
kind: album
artist: Alison Krauss
relationship: RELATED_TO
related_type: ['__KGBuilder__', 'Work', '__Entity__']
related_work_or_entity: So Long So Wrong
</item 13>

<item 14>
Graph query row:
album: Now That I've Found You: A Collection
release_date: February 7, 1995
kind: album
artist: Alison Krauss
relationship: RELATED_TO
related_type: ['__KGBuilder__', '__Entity__', 'Organization']
related_work_or_entity: Rounder Records
</item 14>

<item 15>
Graph query row:
album: Now That I've Found You: A Collection
release_date: February 7, 1995
kind: album
artist: Alison Krauss
relationship: RELATED_TO
related_type: ['__KGBuilder__', 'Work', '__Entity__']
related_work_or_entity: I Know Who Holds Tomorrow
</item 15>

<item 16>
Graph query row:
album: Now That I've Found You: A Collection
release_date: February 7, 1995
kind: album
artist: Alison Krauss
relationship: RELATED_TO
related_type: ['__KGBuilder__', '__Entity__', 'Concept']
related_work_or_entity: U.S. Billboard Top Country Albums
</item 16>

<item 17>
Graph query row:
album: Now That I've Found You: A Collection
release_date: February 7, 1995
kind: album
artist: Alison Krauss
relationship: RELATED_TO
related_type: ['__KGBuilder__', '__Entity__', 'Concept']
related_work_or_entity: Country
</item 17>
=== END ITEM id=68b1b54a64f42944 ===

=== ITEM id=aabf460debfa5210 ===
Gold sentences and facts:
Sentence 0: The earliest predecessors of the eye were photoreceptor proteins that sense light , found even in unicellular organisms , called `` eyespots '' . Eyespots can only sense ambient brightness : they can distinguish light from dark , sufficient for photoperiodism and daily synchronization of circadian rhythms . They are insufficient for vision , as they can not distinguish shapes or determine the direction light is coming from . Eyespots are found in nearly all major animal groups , and are common among unicellular organisms , including euglena . The euglena 's eyespot , called a stigma , is located at its anterior end . It is a small splotch of red pigment which shades a collection of light sensitive crystals . Together with the leading flagellum , the eyespot allows the organism to move in response to light , often toward the light to assist in photosynthesis , and to predict day and night , the primary function of circadian rhythms . Visual pigments are located in the brains of more complex organisms , and are thought to have a role in synchronising spawning with lunar cycles . By detecting the subtle changes in night - time illumination , organisms could synchronise the release of sperm and eggs to maximise the probability of fertilisation .
  Fact 0.0: The earliest predecessors of the eye were photoreceptor proteins that sense light.
  Fact 0.1: The earliest predecessors of the eye were found even in unicellular organisms.
  Fact 0.2: The earliest predecessors of the eye are called 'eyespots'.
  Fact 0.3: Eyespots can only sense ambient brightness.
  Fact 0.4: Eyespots can distinguish light from dark.
  Fact 0.5: Eyespots are sufficient for photoperiodism.
  Fact 0.6: Eyespots are sufficient for daily synchronization of circadian rhythms.
  Fact 0.7: Eyespots are insufficient for vision.
  Fact 0.8: Eyespots cannot distinguish shapes.
  Fact 0.9: Eyespots cannot determine the direction light is coming from.
  Fact 0.10: Eyespots are found in nearly all major animal groups.
  Fact 0.11: Eyespots are common among unicellular organisms.
  Fact 0.12: Euglena is a unicellular organism that has an eyespot.
  Fact 0.13: The euglena's eyespot is called a stigma.
  Fact 0.14: The euglena's stigma is located at its anterior end.
  Fact 0.15: The euglena's stigma is a small splotch of red pigment.
  Fact 0.16: The euglena's stigma shades a collection of light sensitive crystals.
  Fact 0.17: Together with the leading flagellum, the euglena's eyespot allows the organism to move in response to light.
  Fact 0.18: The euglena often moves toward the light to assist in photosynthesis.
  Fact 0.19: The euglena's eyespot allows the organism to predict day and night.
  Fact 0.20: Predicting day and night is the primary function of circadian rhythms.
  Fact 0.21: Visual pigments are located in the brains of more complex organisms.
  Fact 0.22: Visual pigments are thought to have a role in synchronising spawning with lunar cycles.
  Fact 0.23: By detecting subtle changes in night-time illumination, organisms could synchronise the release of sperm and eggs.
  Fact 0.24: Synchronising the release of sperm and eggs is meant to maximise the probability of fertilisation.

Retrieved items:
<item 1>
Evolution of the eye
149 Jump up ^ Lee , M.S.Y. ; Jago , J.B. ; Garcia - Bellido , D.C. ; Edgecombe , G.E. ; Gehling , J. G ; Paterson , J.R. ( 2011 ) . `` Modern optics in
exceptionally preserved eyes of Early Cambrian arthropods from Australia '' . Nature . 474 : 631 -- 634 . doi : 10.1038 / nature10097 . PMID 21720369 . ^ Jump
up to : Darwin , Charles ( 1859 ) . On the Origin of Species . London : John Murray . Jump up ^ Gehring WJ ( 2005 ) . `` New perspectives on eye development and
the evolution of eyes and photoreceptors '' . J. Hered. 96 ( 3 ) : 171 -- 84 . doi : 10.1093 / jhered / esi027 . PMID 15653558 . Jump up ^ Nilsson , D. - E. (
2013 ) . `` Eye evolution and its functional basis '' . Visual Neuroscience.
</item 1>

<item 2>
Evolution of the eye
`` Eye ancestry : old genes for new eyes '' . Current Biology . 6 ( 1 ) : 39 -- 42 . doi : 10.1016 / S0960 - 9822 ( 02 ) 00417 - 7 . PMID 8805210 . Jump up ^
Zinovieva , R. ; Piatigorsky , J. ; Tomarev , S.I. ( 1999 ) . `` O - Crystallin , arginine kinase and ferritin from the octopus lens '' . Biochimica et
Biophysica Acta ( BBA ) - Protein Structure and Molecular Enzymology. 1431 ( 2 ) : 512 -- 517 . doi : 10.1016 / S0167 - 4838 ( 99 ) 00066 - 7 . ^ Jump up to :
Scotland , R.W. ( 2010 ) . `` Deep homology : A view from systematics '' . BioEssays. 32 ( 5 ) : 438 -- 449 . doi : 10.1002 / bies.
</item 2>

<item 3>
Graph query row:
entity: photoreceptor cells
relationship: RELATED_TO
connected: eyes
</item 3>

<item 4>
Graph query row:
entity: photoreceptor cells
relationship: RELATED_TO
connected: Pax 6 genes
</item 4>

<item 5>
Graph query row:
entity: photoreceptor cells
relationship: RELATED_TO
connected: Cambrian explosion
</item 5>

<item 6>
Evolution of the eye
Jump up ^ Dawkins , Richard ( 1986 ) . The Blind Watchmaker . ^ Jump up to : Schoenemann , B. ; Liu , J.N. ; Shu , D.G. ; Han , J. ; Zhang , Z.F. ( 2008 ) . ``
A miniscule optimized visual system in the Lower Cambrian '' . Lethaia. 42 ( 3 ) : 265 -- 273 . doi : 10.1111 / j. 1502 - 3931.2008. 00138. x . Jump up ^ Ali ,
M.A. and M.A. Klyne. 1985 . Vision in vertebrates . New York : Plenum Press Jump up ^ Fernald , Russell D. ( 2001 ) . The Evolution of Eyes : Why Do We See What
We See ? Karger Gazette 64 : `` The Eye in Focus '' . Jump up ^ Fernald , Russell D. ( 1998 ) . Aquatic Adaptations in Fish Eyes . New York , Springer .
</item 6>

<item 7>
Evolution of the eye
Oxford Journals. 96 ( 3 ) : 171 -- 184 . doi : 10.1093 / jhered / esi027 . PMID 15653558 . Retrieved 2008 - 04 - 26 . ^ Jump up to : MF Land ; RD Fernald ( 1992
) . `` The Evolution of Eyes '' . Annual Review of Neuroscience. 15 : 1 -- 29 . doi : 10.1146 / annurev. ne. 15.030192. 000245 . PMID 1575438 . Jump up ^ Autrum
, H ( 1979 ) . `` Introduction '' . In H. Autrum . Comparative Physiology and Evolution of Vision in Invertebrates - A : Invertebrate Photoreceptors . Handbook
of Sensory Physiology . VII / 6A . New York : Springer - Verlag . pp. 6 -- 9 . ISBN 3 - 540 - 08837 - 7 .
</item 7>

<item 8>
Graph query row:
entity: photoreceptor cells
relationship: RELATED_TO
connected: chemoreceptor cells
</item 8>

<item 9>
Evolution of the eye
PMID 25734540 . Greuet , C ( 1968 ) . `` Organisation ultrastructurale de l'ocelle de deux Peridiniens Warnowiidae , Erythropsis pavillardi Kofoid et Swezy et
Warnowia pulchra Schiller '' . Protistologica. 4 : 209 -- 230 . Gregory S. Gavelis , Shiho Hayakawa , Richard A. White III , Takashi Gojobori , Curtis A. Suttle
, Patrick J. Keeling , Brian S. Leander ( 2015 ) . `` Eye - like ocelloids are built from different endosymbiotically acquired components '' . Nature . 523 :
204 -- 7 . Bibcode : 2015Natur. 523 ... 204G . doi : 10.1038 / nature14593 . PMID 26131935 . CS1 maint : Uses authors parameter ( link ) Oakley , Todd H. ;
Speiser , Daniel I. ( 2015 ) . `` How Complexity Originates : The Evolution of Animal Eyes '' .
</item 9>

<item 10>
Evolution of the eye
This suggests that when the two lineages diverged in the Precambrian , they had only very primitive light receptors , which developed into more complex eyes
independently . Early eyes ( edit ) The basic light - processing unit of eyes is the photoreceptor cell , a specialized cell containing two types of molecules
in a membrane : the opsin , a light - sensitive protein , surrounding the chromophore , a pigment that distinguishes colors . Groups of such cells are termed ``
eyespots '' , and have evolved independently somewhere between 40 and 65 times . These eyespots permit animals to gain only a very basic sense of the direction
and intensity of light , but not enough to discriminate an object from its surroundings . Developing an optical system that can discriminate the direction of
light to within a few degrees is apparently much more difficult , and only six of the thirty - some phyla possess such a system . However , these phyla account
for 96 % of living species . The planarian has `` cup '' eyespots that can slightly distinguish light direction .
</item 10>

<item 11>
Evolution of the eye
1 ( 4 ) : 439 -- 447 . doi : 10.1007 / s12052 - 008 - 0084 - 1 . ISSN 1936 - 6426 . Further reading ( edit ) Lamb TD , Collin SP , Pugh EN ( December 2007 ) .
`` Evolution of the vertebrate eye : opsins , photoreceptors , retina and eye cup '' . Nat. Rev. Neurosci. 8 ( 12 ) : 960 -- 76 . doi : 10.1038 / nrn2283 . PMC
3143066 . PMID 18026166 . Illustration . Review Lamb , TD ( 2011 ) . `` Evolution of the Eye '' ( PDF ) . Scientific American . 305 : 64 -- 69 . doi : 10.1038 /
scientificamerican0711 - 64 . Land , Michael F. ; Nilsson , Dan - Eric ( 2012 ) . `` The origin of vision '' . Animal Eyes ( 2 ed . ) . Oxford : Oxford
University Press .
</item 11>

<item 12>
Evolution of the eye
Evolution of the eye - wikipedia Evolution of the eye Jump to : navigation , search Major stages in the evolution of the eye in vertebrates . The evolution of
the eye is attractive to study , because the eye distinctively exemplifies an analogous organ found in many animal forms . Complex , image - forming eyes have
evolved independently several times . The evolutionary geneticist Walter Gering holds a different opinion . According his theory , the genes engaged with
photoreceptor cells ( so called Pax 6 genes ) appeared probably at once in the life history , at a very early stage of the evolution . Probably , the complex ,
image forming kinds of eyes are at least five : Chidaria - lens eye , complex eye ; Molluscs - lens eye , everse retina ; Annelids - lens eye , everse retina ;
Arthropods - facet eye ; Vertebrate - lens eye , inverse retina Complex eyes appeared first within the few million years of the Cambrian explosion . From before
the Cambrian , no evidence of eyes has survived , but diverse eyes are known from the Burgess shale of the Middle Cambrian , and from the slightly older Emu Bay
Shale .
</item 12>

<item 13>
Evolution of the eye
`` Eye evolution and its functional basis '' . Visual Neuroscience. 30 : 5 -- 20 . doi : 10.1017 / s0952523813000035 . ^ Jump up to : Nilsson , D. - E. ; Pelger
, S. ( 1994 ) . `` A pessimistic estimate of the time required for an eye to evolve '' . Proceedings of the Royal Society B : Biological Sciences . 256 : 53 --
58 . doi : 10.1098 / rspb. 1994.0048 . PMID 8008757 . Jump up ^ Young , G.C. ( 2008 ) . `` Early evolution of the vertebrate eye -- fossil evidence '' . Evo Edu
Outreach. 1 : 427 -- 438 . doi : 10.1007 / s12052 - 008 - 0087 - y . Jump up ^ Parker , A.R. ( 2009 ) . `` On the origin of optics '' . Optics & Laser
Technology . 43 ( 2 ) : 323 -- 329 .
</item 13>

<item 14>
Evolution of the eye
PMID 7892602 . Jump up ^ Tomarev , S.I. ; Callaerts , P. ; Kos , L. ; Zinovieva , R. ; Halder , G. ; Gehring , W. ; Piatigorsky , J. ( 1997 ) . `` Squid Pax - 6
and eye development '' . Proceedings of the National Academy of Sciences of the United States of America . 94 ( 6 ) : 2421 -- 2426 . Bibcode : 1997PNAS ...
94.2421 T . doi : 10.1073 / pnas. 94.6. 2421 . PMC 20103 . PMID 9122210 . ^ Jump up to : Gehring , W.J. ( 13 January 2005 ) . `` New Perspectives on Eye
Development and the Evolution of Eyes and Photoreceptors '' ( Full text ) . Journal of Heredity . Oxford Journals. 96 ( 3 ) : 171 -- 184 .
</item 14>

<item 15>
Graph query row:
entity: photoreceptor cells
relationship: RELATED_TO
connected: vertebrate eye
</item 15>
=== END ITEM id=aabf460debfa5210 ===

=== ITEM id=f82a381dcb970121 ===
Gold sentences and facts:
Sentence 0: The Curse of Oak Island follows brothers Marty and Rick Lagina , originally from Kingsford , Michigan , through their efforts to find the speculated treasure or historical artifacts believed to be on Oak Island . The series discusses the history of the island , recent discoveries , theories , and prior attempts to investigate the site . Areas of interest include the `` Money Pit '' , Borehole 10 - x , Smith 's Cove , `` Nolan 's Cross '' , the `` Hatch '' , the `` Watchtower '' and the `` Swamp '' .
  Fact 0.0: The Curse of Oak Island follows brothers Marty and Rick Lagina.
  Fact 0.1: Marty and Rick Lagina are originally from Kingsford, Michigan.
  Fact 0.2: The Curse of Oak Island follows the Laginas through their efforts to find the speculated treasure or historical artifacts believed to be on Oak Island.
  Fact 0.3: The speculated treasure or historical artifacts are believed to be on Oak Island.
  Fact 0.4: The Curse of Oak Island series discusses the history of the island.
  Fact 0.5: The series discusses recent discoveries on the island.
  Fact 0.6: The series discusses theories about the island.
  Fact 0.7: The series discusses prior attempts to investigate the site.
  Fact 0.8: Areas of interest include the 'Money Pit'.
  Fact 0.9: Areas of interest include Borehole 10-x.
  Fact 0.10: Areas of interest include Smith's Cove.
  Fact 0.11: Areas of interest include 'Nolan's Cross'.
  Fact 0.12: Areas of interest include the 'Hatch'.
  Fact 0.13: Areas of interest include the 'Watchtower'.
  Fact 0.14: Areas of interest include the 'Swamp'.

Retrieved items:
<item 1>
Graph query row:
subject: The Curse of Oak Island
relationship: LOCATED_IN
location: Nova Scotia
</item 1>

<item 2>
Graph query row:
subject: Oak Island swamp
relationship: LOCATED_IN
location: Oak Island
</item 2>

<item 3>
The Curse of Oak Island
overall No. in season Title Original air date Viewers ( millions ) 29 `` Going for Broke '' November 15 , 2016 ( 2016 - 11 - 15 ) 2.62 The team agrees to turn
focus over to the Money Pit and review what they have collected . Researcher Zena Halpern presents a handmade copy of what she claims is a French map from 1347
of Oak Island . The map shows locations for a `` hatch '' , a `` valve '' and an `` anchor '' on the Island . A theory is presented based on a purported
decoding of a cipher that Joab at the time of King David transported the Ark of the Covenant to North Africa from which it was later transported onward by the
Knights Templar from the then - Portuguese town of Safi in Morocco to Oak Island . 30 `` Always Forward '' November 22 , 2016 ( 2016 - 11 - 22 ) 2.66 While
preparing to excavate the Money Pit , the team investigates an 18th century well on mainland Nova Scotia .
</item 3>

<item 4>
The Curse of Oak Island
S. Wild West Tech The XY Factor Upcoming Blue Book The Commanders Retrieved from ``
https://en.wikipedia.org/w/index.php?title=The_Curse_of_Oak_Island&oldid=832882458 '' Categories : 2014 American television series debuts History ( U.S. TV
channel ) original programs English - language television programs 2010s American television series Television shows set in Nova Scotia Oak Island Hidden
categories : Use mdy dates from August 2014 Talk Contents About Wikipedia Français Latviešu Română Русский Edit links This page was last edited on 28 March 2018
, at 14 : 10 . About Wikipedia
</item 4>

<item 5>
The Curse of Oak Island
2018 '' . Showbuzz Daily . Retrieved March 7 , 2018 . Further reading ( edit ) MacDonald , David ( 1965 ) . `` Treasure Hunt : The Mystery of Oak Island '' .
Reader 's Digest -- via readersdigest.ca .
</item 5>

<item 6>
The Curse of Oak Island
Series overview ( edit ) Season Episodes Originally aired First aired Last aired 5 January 5 , 2014 ( 2014 - 01 - 05 ) February 9 , 2014 ( 2014 - 02 - 09 ) 10
November 4 , 2014 ( 2014 - 11 - 04 ) January 13 , 2015 ( 2015 - 01 - 13 ) 13 November 10 , 2015 ( 2015 - 11 - 10 ) February 2 , 2016 ( 2016 - 02 - 02 ) 16
November 15 , 2016 ( 2016 - 11 - 15 ) February 21 , 2017 ( 2017 - 02 - 21 ) 5 18 November 7 , 2017 ( 2017 - 11 - 07 ) March 6 , 2018 ( 2018 - 03 - 06 )
Production ( edit ) Having started in 2014 , The Curse of Oak Island is currently in its fifth season . On August 2 , 2016 , it was announced that Nova Scotia
Business Inc.
</item 6>

<item 7>
The Curse of Oak Island
Background ( edit ) The Lagina brothers became fascinated with the island after reading the January 1965 issue of Reader 's Digest magazine that features an
article on the Restall family 's work to investigate the so - called `` Money Pit '' . Marty and his brother Rick obtained a controlling interest in Oak Island
Tours , which reportedly owns most of the island . The brothers were later approached by Prometheus Entertainment to do a reality show . Rick and Marty have
engaged the assistance of father - and - son Dan and Dave Blankenship , permanent residents of the island who have likewise been searching for treasure since
the 1960s . Theories ( edit ) The series explores various Oak Island theories and conjectures through conversations with independent researchers .
</item 7>

<item 8>
The Curse of Oak Island
Contents ( hide ) 1 Overview 1.1 Background 1.2 Theories 2 Series overview 3 Production 4 Episodes 4.1 Season 1 ( 2014 ) 4.2 Season 2 ( 2014 -- 2015 ) 4.3
Season 3 ( 2015 -- 2016 ) 4.4 Season 4 ( 2016 -- 2017 ) 4.5 Season 5 ( 2017 -- 2018 ) 5 References 6 Further reading 7 External links Overview ( edit ) The
Curse of Oak Island follows brothers Marty and Rick Lagina , originally from Kingsford , Michigan , through their efforts to find the speculated treasure or
historical artifacts believed to be on Oak Island . The series discusses the history of the island , recent discoveries , theories , and prior attempts to
investigate the site . Areas of interest include the `` Money Pit '' , Borehole 10 - x , Smith 's Cove , `` Nolan 's Cross '' , the `` Hatch '' , the ``
Watchtower '' and the `` Swamp '' .
</item 8>

<item 9>
The Curse of Oak Island
On August 2 , 2016 , it was announced that Nova Scotia Business Inc. approved $1,271,546 in film funding for the production of the 4th season . In October 2017
, a fifth season was announced , which began airing on November 7 , 2017 . Episodes ( edit ) Season 1 ( 2014 ) ( edit ) No . overall No. in season Title
Original air date Viewers ( millions ) `` What Lies Below '' January 5 , 2014 ( 2014 - 01 - 05 ) 2.53 Brothers Rick and Marty Lagina now own most of Oak Island
. Determined to resume the search for treasure on the island , they explore an abandoned shaft called Borehole 10 - X and use water pressure to pump material to
the surface . They find a tiny piece of metal .
</item 9>

<item 10>
The Curse of Oak Island
61 17 `` A Family Album '' February 27 , 2018 ( 2018 - 02 - 27 ) 1.98 The episode includes accounts of activities on the Oak Island by the McGinnis family ,
descendants of Samuel Ball , William Chappell 's family , the Restall Family and the family of Robert Dunfield . The Blankenship / Nolan disputes on Oak Island
are discussed . The episode ends with a short history of the Lagina brothers ' interest in Oak Island . Note : This episode is a special . 62 18 `` Amazing
Discoveries '' March 6 , 2018 ( 2018 - 03 - 06 ) 2.89 The gem in the Jewelry fragment found on the island is a Rhodolite Garnet stone of small value . The
drilling obstruction in the `` money pit '' borehole is determined by divers to be a granite boulder . References ( edit ) ^ Jump up to : `` Shows A-Z - curse
of oak island , the on history '' . The Futon Critic . Retrieved December 6 , 2017 .
</item 10>

<item 11>
The Curse of Oak Island
History Channel . Retrieved January 4 , 2014 . Jump up ^ Matthes , Ilsa ( February 8 , 2014 ) . `` Treasure or Money Pit ? : History Channel Star Visits
Escanaba School '' . Daily Press . Escanaba , Michigan . Jump up ^ Conaton , Chris ( January 5 , 2014 ) . `` ' The Curse of Oak Island ' : Slow Going '' .
PopMatters . Retrieved January 6 , 2014 . Jump up ^ `` History Channel 's Curse of Oak Island gets film funding for Season 4 '' . The Chronicle Herald . 2016 -
08 - 02 . Retrieved 2016 - 08 - 02 . Jump up ^ Facebook , `` Curse of Oak Island '' . Accessed October 6 , 2017 . Jump up ^ `` Sunday Cable Ratings : ' Real
Housewives of Atlanta ' Wins Night , ' Blood , Sweat & Heels ' , ' Ax Men ' , ' Thicker Than Water ' & More - TV By The Numbers by zap2it.com '' .
</item 11>

<item 12>
The Curse of Oak Island
The Curse of Oak Island - wikipedia The Curse of Oak Island Jump to : navigation , search The Curse of Oak Island Genre Reality Narrated by Robert Clotworthy
Country of origin Canada Original language ( s ) English No. of seasons 5 No. of episodes 62 Production Executive producer ( s ) Kevin Burns Camera setup
Multiple Running time 43 minutes Production company ( s ) Prometheus Entertainment Distributor A&E Networks Release Original network History Canada Original
release January 5 , 2014 ( 2014 - 01 - 05 ) -- present External links Website Production website The Curse of Oak Island is an active reality television series
that first premiered in Canada on the History network on January 5 , 2014 . The show features what is known as the Oak Island mystery , showing efforts to
search for historical artifacts and treasure .
</item 12>
=== END ITEM id=f82a381dcb970121 ===
