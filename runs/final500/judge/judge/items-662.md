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

=== ITEM id=d3fffd21adadbe4c ===
Gold sentences and facts:
Sentence 0: In negotiation theory , the best alternative to a negotiated agreement or BATNA is the most advantageous alternative course of action a party can take if negotiations fail and an agreement can not be reached . BATNA is the key focus and the driving force behind a successful negotiator . A party should generally not accept a worse resolution than its BATNA . Care should be taken , however , to ensure that deals are accurately valued , taking into account all considerations , such as relationship value , time value of money and the likelihood that the other party will live up to their side of the bargain . These other considerations are often difficult to value , since they are frequently based on uncertain or qualitative considerations , rather than easily measurable and quantifiable factors .
  Fact 0.0: In negotiation theory, the best alternative to a negotiated agreement (BATNA) is the most advantageous course of action a party can take if negotiations fail and no agreement can be reached.
  Fact 0.1: BATNA is the key focus of a successful negotiator.
  Fact 0.2: BATNA is the driving force behind a successful negotiator.
  Fact 0.3: A party should generally not accept a resolution worse than its BATNA.
  Fact 0.4: Care should be taken to ensure that deals are accurately valued.
  Fact 0.5: Deal valuation should take into account all considerations.
  Fact 0.6: Relationship value is a consideration in valuing a deal.
  Fact 0.7: Time value of money is a consideration in valuing a deal.
  Fact 0.8: The likelihood that the other party will live up to its side of the bargain is a consideration in valuing a deal.
  Fact 0.9: The other considerations are often difficult to value.
  Fact 0.10: The other considerations are frequently based on uncertain or qualitative considerations.
  Fact 0.11: The other considerations are qualitative rather than easily measurable and quantifiable factors.

Retrieved items:
<item 1>
Best alternative to a negotiated agreement
( 2004 ) . `` Cultural Differences and Cognitive Dynamics : Expanding the Cognitive Perspective on Negotiation '' . In Gelfand , Michele J. ; Brett , Jeanne M.
The Handbook of Negotiation and Culture . Stanford University Press . pp. 45 -- 70 . ISBN 978 - 0 - 8047 - 4586 - 4 . Jump up ^ Myerson , Roger B. `` Nash
Equilibrium and the History of Economic Theory '' ( PDF ) . Nash Equilibrium and the History of Economic Theory . Journal of Economic Literature . Retrieved 1
October 2012 . Jump up ^ Hawkins and Steiner , Jeff and Neil . `` The Nash Equilibrium Meets Batna '' ( PDF ) . Gamed Therory Varied Uses in ADR . Harvard
University Press . Archived from the original ( PDF ) on February 28 , 2013 . Retrieved 1 October 2012 .
</item 1>

<item 2>
Bond (finance)
An alternative process for bond issuance , which is commonly used for smaller issues and avoids this cost , is the private placement bond . Bonds sold directly
to buyers may not be tradeable in the bond market . Historically an alternative practice of issuance was for the borrowing government authority to issue bonds
over a period of time , usually at a fixed price , with volumes sold on a particular day dependent on market conditions . This was called a tap issue or bond
tap . Features ( edit ) Principal ( edit ) 1978 $1,000 U.S. Treasury Bond Nominal , principal , par , or face amount is the amount on which the issuer pays
interest , and which , most commonly , has to be repaid at the end of the term . Some structured bonds can have a redemption amount which is different from the
face amount and can be linked to performance of particular assets . Maturity ( edit ) The issuer has to repay the nominal amount on the maturity date . As long
as all due payments have been made , the issuer has no further obligations to the bond holders after the maturity date .
</item 2>

<item 3>
Best alternative to a negotiated agreement
Contents ( hide ) 1 Overview 2 History 3 See also 4 References 5 External links Overview ( edit ) The BATNA is often seen by negotiators not as a safety net ,
but rather as a point of leverage in negotiations . Although a negotiator 's alternative options should , in theory , be straightforward to evaluate , the
effort to understand which alternative represents a party 's BATNA is often not invested . Options need to be actual and actionable to be of value , however
without the investment of time , options will frequently be included that fail on one of these criteria . Most managers overestimate their BATNA whilst
simultaneously investing too little time into researching their real options . This can result in poor or faulty decision making and negotiating result .
Negotiators also need to be aware of the other negotiator 's BATNA and to identify how it compares to what they are offering . Some people may adopt aggressive
, coercive , threatening and / or deceptive techniques , this is known as a hard negotiation style ; a theoretical example of this is adversarial approach style
negotiation . Others may employ a soft style , which is friendly , trusting , compromising , and conflict avoiding .
</item 3>

<item 4>
Best alternative to a negotiated agreement
See also ( edit ) Conflict resolution research Opportunity cost References ( edit ) Jump up ^ Best Alternative to a Negotiated Agreement Negotiation Experts
Jump up ^ Best Alternative to a Negotiated Agreement YArooms Jump up ^ Negotiation , Readings , Exercises and Cases , Roy J. Lewicki ^ Jump up to : Nolan -
Haley , Jaqueline M ( 2001 ) . Alternative Dispute Resolution in a Nutshell . Thomson West . pp. 39 -- 50 . ISBN 978 - 0 - 314 - 18014 - 8 . ^ Jump up to :
Fisher and Ury , Roger and William ( 2011 ) . Getting to YES . Penguin Books . pp. 1 -- 170 . ISBN 978 - 0 - 14 - 311875 - 6 . Jump up ^ Gulliver , P.H ( 1979 )
. Disputes and Negotiation : A Cross Culture Perspective . Academic Press . p. 287 . Jump up ^ Morris , Michael W. ; Gelfand , Michele J. ( 2004 ) .
</item 4>

<item 5>
Best alternative to a negotiated agreement
Others may employ a soft style , which is friendly , trusting , compromising , and conflict avoiding . According to Fisher and Ury , when hard negotiators meet
soft negotiators , the hard negotiators usually win their position , but at the cost of potentially damaging the long term relationship between the parties .
Attractive alternatives are needed to develop a strong BATNA . In the best - selling book Getting to YES : Negotiating Agreement Without Giving In , the authors
give 3 suggestions of how to accomplish this : Inventing a list of actions one might take if no agreement is reached Converting some of the more promising ideas
and transforming them into tangible and partial alternatives Selecting the alternative that sounds best In negotiations involving different cultures , all
parties need to account for cultural cognitive behaviors and should not let judgments and biases affect the negotiation . The individual should be separate from
the objective . The purpose here , as Gulliver mentions , is for negotiation parties to be aware . Preparation at all levels , including prejudice - free
thoughts , emotion - free behavior , bias - free behavior are helpful according to Morris and Gelfand .
</item 5>

<item 6>
Best alternative to a negotiated agreement
History ( edit ) BATNA was developed by negotiation researchers Roger Fisher and William Ury of the Harvard Program on Negotiation ( PON ) , in their series of
books on Principled negotiation that started with Getting to YES , unwittingly duplicating the game theory concept of a disagreement point from bargaining
problems pioneered by Nobel Laureate John Forbes Nash decades earlier . A Nash Equilibrium is reached among a group of players when no player can benefit from
changing strategies if every other player sticks to their current strategy . For example , Amy and Phil are in Nash Equilibrium if Amy is making the best
decision she can , taking into account Phil 's decision , and Phil is making the best decision he can , taking into account Amy 's decision . Likewise , a group
of players are in Nash Equilibrium if each one is making the best decision that he or she can , taking into account the decisions of the others .
</item 6>

<item 7>
Best alternative to a negotiated agreement
Retrieved 1 October 2012 . External links ( edit ) BATNA Explained by David Venter Alternative definition of BATNA Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Best_alternative_to_a_negotiated_agreement&oldid=819966880 '' Categories : Negotiation Dispute resolution Business
terms Hidden categories : Articles needing more detailed references Articles needing additional references from July 2013 All articles needing additional
references All articles lacking reliable references Articles lacking reliable references from July 2013 All articles with unsourced statements Articles with
unsourced statements from October 2007 Articles lacking reliable references from February 2015 Wikipedia articles needing page number citations from July 2013
Talk Contents About Wikipedia Deutsch Español Հայերեն Polski Edit links This page was last edited on 12 January 2018 , at 08 : 19 . Text is available under the
Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy .
Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization .
</item 7>

<item 8>
Best alternative to a negotiated agreement
, a non-profit organization . About Wikipedia
</item 8>

<item 9>
Best alternative to a negotiated agreement
Best alternative to a negotiated agreement - wikipedia Best alternative to a negotiated agreement Jump to : navigation , search This article needs additional
citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . (
July 2013 ) ( Learn how and when to remove this template message ) In negotiation theory , the best alternative to a negotiated agreement or BATNA is the most
advantageous alternative course of action a party can take if negotiations fail and an agreement can not be reached . BATNA is the key focus and the driving
force behind a successful negotiator . A party should generally not accept a worse resolution than its BATNA . Care should be taken , however , to ensure that
deals are accurately valued , taking into account all considerations , such as relationship value , time value of money and the likelihood that the other party
will live up to their side of the bargain . These other considerations are often difficult to value , since they are frequently based on uncertain or
qualitative considerations , rather than easily measurable and quantifiable factors .
</item 9>

<item 10>
Board of directors
This does not mean , however , that the board can not agree to the company entering into a contract which binds the company to a certain course , even if
certain actions in that course will require further board approval . The company remains bound , but the directors retain the discretion to vote against taking
the future actions ( although that may involve a breach by the company of the contract that the board previously approved ) . `` Conflict of duty and interest
'' ( edit ) As fiduciaries , the directors may not put themselves in a position where their interests and duties conflict with the duties that they owe to the
company . The law takes the view that good faith must not only be done , but must be manifestly seen to be done , and zealously patrols the conduct of directors
in this regard ; and will not allow directors to escape liability by asserting that his decision was in fact well founded . Traditionally , the law has divided
conflicts of duty and interest into three sub-categories .
</item 10>
=== END ITEM id=d3fffd21adadbe4c ===

=== ITEM id=bd50d2ce31f34f32 ===
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
Around that time , Spanish forces in Cuba numbered about 80,000 , of which 20,000 were regular troops and 60,000 were Spanish and Cuban volunteer militia . The
latter were a locally enlisted force that took care of most of the `` guard and police '' duties on the island . Wealthy landowners would `` volunteer '' a
number of their slaves to serve in this force , which was under local control as militia and not under official military command . By December , Spain had sent
98,412 regular troops to the island , and the colonial government increased the Volunteer Corps to 63,000 men . By the end of 1897 , there were 240,000 regulars
and 60,000 irregulars on the island . The revolutionaries were far outnumbered . The Mambises were named after the Negro Spanish officer , Juan Ethninius Mamby
who joined the Dominican fight for independence in 1844 . The Spanish soldiers referred to the insurgents in the Dominican Republic as `` the men of Mamby '' or
`` Mambies '' .
</item 1>

<item 2>
In mid-September, representatives of Cuba’s five Liberation Army Corps adopted the Jimaguayú Constitution, creating a Government Council led by Salvador
Cisneros and Bartolomé Masó. Rebel forces outmaneuvered Spanish troops and crossed Martínez-Campos’s fortified Júcaro–Morón trocha, designed to contain them in
eastern Cuba. They invaded all provinces, reaching the island’s western tip on January 22, 1896. Replaced by Val
</item 2>

<item 3>
Cuban War of Independence
As stipulated at the Jimaguayü Assembly two years earlier , a second Constituent Assembly met in La Yaya , Camagüey , on October 10 , 1897 . The newly adopted
constitution provided that military command was to be subordinated to civilian rule . The government was confirmed , naming Bartolomé Masó President and Domingo
Méndez Capote Vice President . Madrid decided to change its policy towards Cuba , and replaced Weyler . It also drew up a colonial constitution for Cuba and
Puerto Rico , and installed a new government in Havana . But with half the country out of its control and the other half in arms , the colonial government was
powerless and these changes were rejected by the rebels . The Maine incident Main article : USS Maine ( ACR - 1 ) Wreckage of the Maine , 1898 The Cuban
struggle for independence had captured the American imagination for years . Some newspapers had agitated for US intervention , especially because of its large
financial investment , and featured sensational stories of Spanish atrocities against the native Cuban population , which were exaggerated for propaganda . Such
coverage continued after Spain had replaced Weyler and changed its policies .
</item 3>

<item 4>
After slavery ended in 1886, Cuba’s society and economy changed, while José Martí organized exiles and founded the Cuban Revolutionary Party to win independence
without U.S. annexation. Rebellion began in 1895; the 1897 constitution placed military command under civilian rule. The USS Maine exploded in Havana in 1898,
intensifying U.S. public pressure for intervention. Congress backed Cuban independence, disclaimed annexation through the Teller Amendment, and declared war on
Spain in April
</item 4>

<item 5>
Cuban War of Independence
As the war went on , the major obstacle to Cuban success was weapons supply . Although weapons and funding came from within the Cuban exile community and
supporters in the United States , the supply operation violated American laws . These were enforced by the U.S. Coast Guard : of 71 attempted re-supply missions
, only 27 got through ; 5 were stopped by the Spanish , but 33 were stopped by the U.S. Coast Guard . In 1897 , the liberation army maintained a privileged
position in Camagüey and Oriente , where the Spanish controlled only a few cities . Spanish Liberal leader Práxedes Mateo Sagasta admitted in May 1897 : ``
After having sent 200,000 men and shed so much blood , we do n't own more land on the island than what our soldiers are stepping on '' . The rebel force of
3,000 defeated the Spanish in various encounters , such as the Battle of La Reforma , and forcing the surrender of Las Tunas on August 30 . The Spaniards were
kept on the defensive . Las Tunas had been guarded by over 1,000 well - armed and well - supplied men .
</item 5>

<item 6>
Spanish forces in Cuba grew from about 80,000 to 300,000 regular and irregular troops by late 1897, vastly outnumbering the rebels. Insurgents, called Mambises
after a Dominican independence fighter, faced severe weapons shortages and used guerrilla raids, fast horses, and machetes. U.S. enforcement intercepted most
foreign resupply attempts. Despite this, rebels held much of Camagüey and Oriente, defeated Spanish forces, and captured Las Tunas in August
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
Cuba’s 1895–98 independence war, led by José Martí, Gómez, and Maceo, spread from Oriente across the island despite Spanish troop superiority and rebel arms
shortages. Spain’s reconcentration policy caused an estimated 155,000–170,000 civilian deaths. The 1898 USS Maine explosion helped spur U.S. intervention; the
United States defeated Spain, and the Treaty of Paris recognized Cuban independence but excluded Cubans from negotiations. U.S. occupation
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
When Cuba 's first war of independence broke out in 1868 , some of those same soldiers were assigned to Cuba ; they brought along what by then become a
derogatory Spanish slur for insurgents . The Cubans adopted the name with pride . After the Ten Years ' War , the government had forbidden possession of weapons
by private individuals . From the beginning of the uprising , the rebels were hampered by the lack of suitable weapons . They compensated by using guerrilla -
style fighting , based on quick raids and fades to the environment , the element of surprise , mounting their forces on fast horses , and using machetes against
regular troops on the march . They acquired most of their weapons and ammunition in raids on the Spaniards . Between June 11 , 1895 , and November 30 , 1897 ,
out of sixty attempts to bring weapons and supplies to the rebels from outside the country , only one succeeded through the protection of the British . Twenty -
eight ships were intercepted within U.S. territory ; five were intercepted at sea by the U.S.
</item 10>
=== END ITEM id=bd50d2ce31f34f32 ===

=== ITEM id=3cbed2d6864d8fa5 ===
Gold sentences and facts:
Sentence 0: Season Episodes Originally aired First aired Last aired 17 September 13 , 1969 ( 1969 - 09 - 13 ) January 17 , 1970 ( 1970 - 01 - 17 ) 8 September 12 , 1970 ( 1970 - 09 - 12 ) October 31 , 1970 ( 1970 - 10 - 31 ) 16 September 9 , 1978 December 23 , 1978
  Fact 0.0: The table lists 17 episodes with first airing on September 13, 1969 and last airing on January 17, 1970.
  Fact 0.1: The table lists 8 episodes with first airing on September 12, 1970 and last airing on October 31, 1970.
  Fact 0.2: The table lists 16 episodes with first airing on September 9, 1978 and last airing on December 23, 1978.

Retrieved items:
<item 1>
List of Scooby-Doo, Where Are You! episodes
The Willawaw ! '' The Williwaw and Owl Men / Grey Fox and henchmen September 9 , 1978 ( 1978 - 09 - 09 ) While visiting Velma 's Uncle Dave , the gang find out
he has been kidnapped from his home , and that a mysterious Williwaw is haunting the area . 27 `` A Creepy Tangle in the Bermuda Triangle '' Skeleton Men / Dr.
Grimsley September 16 , 1978 ( 1978 - 09 - 16 ) While on a boat flowing down the Gulf Stream , the gang end up in the Bermuda Triangle and discover an airplane
- snatching flying saucer and a trio of Skeleton Men . 28 `` A Scary Night with a Snow Beast Fright '' Snow Beast / Dr. Bapiste September 23 , 1978 ( 1978 - 09
- 23 ) After being invited by Professor Krueger at the North Pole , the gang finds out that he has been captured . After they meet an Eskimo named Chief Minook
, he gets captured as well , and the gang attempts to solve the mystery .
</item 1>

<item 2>
List of Scooby-Doo, Where Are You! episodes
21 `` Scooby 's Night with a Frozen Fright '' Caveman / Professor Wayne Seven Days a Week October 3 , 1970 ( 1970 - 10 - 03 ) While out fishing , Shaggy and
Scooby - Doo hook a two - million - year - old caveman frozen in a block of ice , which the gang learns was lost at sea during a violent storm . They take the
caveman to the Oceanland aquarium , but when someone melts the ice block and the caveman goes on the rampage , the gang decides to stick around and investigate
. 22 5 `` Haunted House Hang - Up '' Headless Spectre / Penrod Stillwall , Phantom / Asa Shanks Love the World October 10 , 1970 ( 1970 - 10 - 10 ) The Mystery
Machine overheats in front of an old spooky mansion , and the gang goes to look for help , but finds out that the house is being haunted by the Headless Spectre
, a terrifying phantom with no head . The mystery gets even stranger when they begin to hear knocking sounds and clues leading to a treasure .
</item 2>

<item 3>
List of Scooby-Doo, Where Are You! episodes
This is initially the series finale and the last episode under the Scooby - Doo , Where Are You ! name until September , 1978 . Season 3 ( 1978 ) ( revival ) (
edit ) Note : Only the first nine episodes of the sixteen episode 1978 season of Scooby - Doo , listed below , were broadcast under the Scooby - Doo , Where are
You ! name . The revival series was canceled afterward , and the remaining seven episodes , that were intended for the revival , were instead broadcast during
the Scooby 's All - Stars block , and all sixteen were rerun during this block and later syndicated as part of The Scooby - Doo Show . See List of The Scooby -
Doo Show episodes for the entire season run and The Scooby - Doo Show for proper production credits . No . overall No. in season Title Villain ( s ) Original
air date 26 `` Watch Out ! The Willawaw ! ''
</item 3>

<item 4>
List of Scooby-Doo, Where Are You! episodes
Wickles September 13 , 1969 ( 1969 - 09 - 13 ) While walking home from the movies , Shaggy and Scooby - Doo discover a black suit of armor in a pickup truck ,
and the gang deliver it to the local museum . The kids later break into the museum to search for clues , after learning that the archaeologist who was
transporting the suit is missing , and they find that the knight 's armor has come to life . `` Hassle in the Castle '' Phantom of Vasquez / Bluestone the Great
September 20 , 1969 ( 1969 - 09 - 20 ) The gang goes out on a boating expedition , but gets lost in a fog and crashes their boat on the mysterious Haunted Isle
. They decide to venture to the abandoned Vázquez Castle on the island for help , but instead encounter a transparent phantom running loose , who threatens them
to leave the island and never return .
</item 4>

<item 5>
List of Scooby-Doo, Where Are You! episodes
banner , as the revival series was cancelled . 35 10 `` The Creepy Creature of Vulture 's Claw '' Mantis Creature / Professor Greer November 11 , 1978 ( 1978 -
11 - 11 ) The gang discovers a humanoid mantis creature roaming around Vulture 's Claw Botanical Gardens . 36 11 `` The Diabolical Disc Demon '' The Phantom /
Ace Decade November 18 , 1978 ( 1978 - 11 - 18 ) The kids discovers the Diabolical Disc Demon . He wants a half - finished song of a missing writer and will
stop at nothing to get it . 37 12 `` Scooby 's Chinese Fortune Kooky Caper '' Moon Monster / Uncle Chin November 25 , 1978 ( 1978 - 11 - 25 ) The gang runs into
The Moon Monster . There are rumors that if you run into his shadow , you will be turned into stone .
</item 5>

<item 6>
List of Scooby-Doo, Where Are You! episodes
( 2015 -- 18 ) Package shows and programming blocks The Scooby - Doo / Dynomutt Hour ( 1976 -- 77 ) Scooby 's All - Star Laff - A-Lympics / Scooby 's All -
Stars ( 1977 -- 78 ) The Richie Rich / Scooby - Doo Show ( 1980 -- 81 ) The Scooby & Scrappy - Doo / Puppy Hour ( 1982 ) Scary Scooby Funnies ( 1984 -- 85 )
Scooby 's Mystery Funhouse ( 1985 -- 86 ) Television films and specials Hanna - Barbera 's All - Star Comedy Ice Revue ( 1978 ) Scooby Goes Hollywood ( 1979 )
Scooby - Doo Meets the Boo Brothers ( 1987 ) Scooby - Doo and the Ghoul School ( 1988 ) Scooby - Doo ! and the Reluctant Werewolf ( 1988 ) Hanna - Barbera 's
50th : A Yabba Dabba Doo Celebration ( 1989 ) Scooby - Doo !
</item 6>

<item 7>
List of Scooby-Doo, Where Are You! episodes
is a Saturday morning cartoon produced by Hanna - Barbera . Contents ( hide ) 1 Series overview 2 Episodes 2.1 Season 1 ( 1969 -- 70 ) 2.2 Season 2 ( 1970 ) 2.3
Season 3 ( 1978 ) ( Revival ) 3 References Series overview ( edit ) Season Episodes Originally aired First aired Last aired 17 September 13 , 1969 ( 1969 - 09 -
13 ) January 17 , 1970 ( 1970 - 01 - 17 ) 8 September 12 , 1970 ( 1970 - 09 - 12 ) October 31 , 1970 ( 1970 - 10 - 31 ) 16 September 9 , 1978 December 23 , 1978
Episodes ( edit ) Season 1 ( 1969 -- 70 ) ( edit ) No . overall No. in season Title Villain ( s ) Original air date `` What a Night for a Knight '' Black Knight
/ Mr.
</item 7>

<item 8>
List of Scooby-Doo, Where Are You! episodes
24 7 `` Who 's Afraid of the Big Bad Werewolf ? '' Ghost of Silas Long the Werewolf / Sheep Rustler Tell Me , Tell Me October 24 , 1970 ( 1970 - 10 - 24 ) Out
on a camping trip , the gang hears the sounds of a wolf , so they decide to follow wolf - like tracks to the open and empty grave of Silas Long , who was
supposedly a werewolf . They follow his tracks further to an old abandoned sawmill filled with wool , empty barrels , and the werewolf ghost himself . 25 8 ``
Do n't Fool with a Phantom '' Wax Phantom / Roger Steven Pretty Mary Sunlight October 31 , 1970 ( 1970 - 10 - 31 ) While participating in the Johnny Sands Dance
Game Show at the local TV station , the teens encounter a phantom made of wax , who steals a safe full of money and kidnaps the station manager . Is the phantom
a hoax , or the result of a curse from an old station employee seeking revenge for the cancellation of his show ?
</item 8>

<item 9>
List of Scooby-Doo, Where Are You! episodes
Blowout Beach Bash ( 2017 ) Scooby - Doo ! & Batman : The Brave and the Bold ( 2018 ) Live - action films Scooby - Doo ( 2002 ) 2 : Monsters Unleashed ( 2004 )
The Mystery Begins ( 2009 ) Curse of the Lake Monster ( 2010 ) Daphne & Velma ( 2018 ) Video games Scooby - Doo ( 1986 ) Mystery ( 1995 ) Mystery of the Fun
Park Phantom ( 1999 ) Mystery Adventures ( 2000 ) Classic Creep Capers ( 2000 ) The Cyber Chase ( 2001 ) Night of 100 Frights ( 2002 ) Mystery Mayhem ( 2004 )
Unmasked ( 2005 ) Who 's Watching Who ?
</item 9>

<item 10>
List of Scooby-Doo, Where Are You! episodes
episodes ( 1969 -- 1970 ) The New Scooby - Doo Movies episodes ( 1972 -- 73 ) The Scooby - Doo Show episodes ( 1976 -- 78 ) Scooby - Doo and Scrappy - Doo (
1979 -- 1980 ) Scooby - Doo and Scrappy - Doo ( 1980 -- 82 ) The New Scooby and Scrappy - Doo Show / The New Scooby - Doo Mysteries ( 1983 -- 84 ) The 13 Ghosts
of Scooby - Doo ( 1985 ) A Pup Named Scooby - Doo episodes ( 1988 -- 1991 ) What 's New , Scooby - Doo ? episodes ( 2002 -- 06 ) Shaggy & Scooby - Doo Get a
Clue ! episodes ( 2006 -- 08 ) Scooby - Doo ! Mystery Incorporated episodes ( 2010 -- 13 ) Be Cool , Scooby - Doo !
</item 10>
=== END ITEM id=3cbed2d6864d8fa5 ===

=== ITEM id=f790408fac52ec61 ===
Gold sentences and facts:
Sentence 0: Operating System Version ( s ) Location Unix , Unix - like , POSIX / etc / hosts Microsoft Windows 3.1 % WinDir % \ HOSTS 95 , 98 , ME % WinDir % \ hosts NT , 2000 , XP , 2003 , Vista , 2008 , 7 , 2012 , 8 , 10 % SystemRoot % \ System32 \ drivers \ etc \ hosts Windows Mobile , Windows Phone Registry key under HKEY_LOCAL_MACHINE \ Comm \ Tcpip \ Hosts Apple Macintosh 9 and earlier Preferences or System folder Mac OS X 10.0 -- 10.1. 5 ( Added through NetInfo or niload ) Mac OS X 10.2 and newer / etc / hosts ( a symbolic link to / private / etc / hosts ) Novell NetWare SYS : etc \ hosts OS / 2 & eComStation `` bootdrive '' : \ mptn \ etc \ Symbian Symbian OS 6.1 -- 9.0 C : \ system \ data \ hosts Symbian OS 9.1 + C : \ private \ 10000882 \ hosts MorphOS NetStack ENVARC : sys / net / hosts AmigaOS < 4 AmiTCP : db / hosts DEVS : Internet / hosts AROS ENVARC : AROSTCP / db / hosts Android / etc / hosts ( a symbolic link to / system / etc / hosts ) iOS iOS 2.0 and newer / etc / hosts ( a symbolic link to / private / etc / hosts ) TOPS - 20 < SYSTEM > HOSTS. TXT Plan 9 / lib / ndb / hosts BeOS / boot / beos / etc / hosts Haiku / boot / common / settings / network / hosts OpenVMS UCX UCX $ HOST TCPware TCPIP $ HOST RISC OS 3.7 , 5 ! Boot. Resources. ! Internet. files. Hosts later boot sequence ! Boot. Choices. Hardware. Disabled. Internet. Files. Hosts
  Fact 0.0: The hosts file on Unix, Unix-like, and POSIX systems is at /etc/hosts
  Fact 0.1: On Microsoft Windows 3.1 the hosts file is at %WinDir%\HOSTS
  Fact 0.2: On Windows 95, 98, and ME the hosts file is at %WinDir%\hosts
  Fact 0.3: On Windows NT, 2000, XP, 2003, Vista, 2008, 7, 2012, 8, and 10 the hosts file is at %SystemRoot%\System32\drivers\etc\hosts
  Fact 0.4: On Windows Mobile and Windows Phone the hosts entry is a registry key under HKEY_LOCAL_MACHINE\Comm\Tcpip\Hosts
  Fact 0.5: On Apple Macintosh 9 and earlier the hosts file is in the Preferences or System folder
  Fact 0.6: On Mac OS X 10.0 to 10.1.5 the hosts file was added through NetInfo or niload
  Fact 0.7: On Mac OS X 10.2 and newer the hosts file is /etc/hosts, a symbolic link to /private/etc/hosts
  Fact 0.8: On Novell NetWare the hosts file is at SYS:etc\hosts
  Fact 0.9: On OS/2 and eComStation the hosts file is at the bootdrive path \mptn\etc\
  Fact 0.10: On Symbian OS 6.1 to 9.0 the hosts file is at C:\system\data\hosts
  Fact 0.11: On Symbian OS 9.1 and later the hosts file is at C:\private\10000882\hosts
  Fact 0.12: On MorphOS (NetStack) the hosts file is at ENVARC:sys/net/hosts
  Fact 0.13: On AmigaOS before 4 the hosts file is at AmiTCP:db/hosts or DEVS:Internet/hosts
  Fact 0.14: On AROS the hosts file is at ENVARC:AROSTCP/db/hosts
  Fact 0.15: On Android the hosts file is /etc/hosts, a symbolic link to /system/etc/hosts
  Fact 0.16: On iOS 2.0 and newer the hosts file is /etc/hosts, a symbolic link to /private/etc/hosts
  Fact 0.17: On TOPS-20 the hosts file is <SYSTEM>HOSTS.TXT
  Fact 0.18: On Plan 9 the hosts file is /lib/ndb/hosts
  Fact 0.19: On BeOS the hosts file is /boot/beos/etc/hosts
  Fact 0.20: On Haiku the hosts file is /boot/common/settings/network/hosts
  Fact 0.21: On OpenVMS with UCX the hosts file is UCX$HOST
  Fact 0.22: On OpenVMS with TCPware the hosts file is TCPIP$HOST
  Fact 0.23: On RISC OS 3.7 and 5 the hosts file is at !Boot.Resources.!Internet.files.Hosts, loaded in the later boot sequence
  Fact 0.24: On RISC OS the disabled hosts file is at !Boot.Choices.Hardware.Disabled.Internet.Files.Hosts

Retrieved items:
<item 1>
hosts (file)
Retrieved November 30 , 2012 . Jump up ^ RISC OS 6.14 Jump up ^ `` Remove Trojan. Qhosts -- Symantec '' . Retrieved May 16 , 2010 . External links ( edit )
Windows TCP / IP in Windows 2000 Professional Microsoft TCP / IP Host Name Resolution Order Blocking Unwanted Connections with a Hosts File ( Creative Commons
License ) Solution if Windows stops you from modifying the Hosts file or the Lmhosts file in Windows Vista / 7 / 8 Beginners Guide on Editing Hosts File List of
Microsoft related domains that Windows wo n't redirect Mac How to block unwanted domains with a HOSTS file on Mac OS X 10.2 or later `` Gas Mask '' - Hosts file
manager for Mac OS X Linux Block unwanted advertisements with / etc / hosts file on Linux Retrieved from `` https://en.wikipedia.org/w/index.
</item 1>

<item 2>
hosts (file)
History of BIND . Retrieved 2017 - 07 - 01 . Jump up ^ `` Cisco Networking Academy Program : First - Year Companion Guide '' , Cisco Systems , Inc. , 2002 ( 2nd
Edition ) , page 676 , ISBN 1 - 58713 - 025 - 4 Jump up ^ `` Linux Network Administrators Guide : Writing hosts and networks files '' . Retrieved May 16 , 2010
. Jump up ^ `` Hosts File '' . Retrieved August 10 , 2011 . Jump up ^ `` Microsoft KB Q314053 : TCP / IP and NBT configuration parameters for Windows XP '' .
Retrieved August 28 , 2010 . Jump up ^ `` Microsoft KB 972034 Revision 2.0 : default hosts files '' . Retrieved August 28 , 2010 . ^ Jump up to : `` Mac OS X :
How to Add Hosts to Local Hosts File '' . Retrieved August 28 , 2010 . ^ Jump up to : `` The Haiku / BeOS Tip Server '' . Retrieved November 30 , 2012 .
</item 2>

<item 3>
hosts (file)
Internet resource blocking Entries in the hosts file may be used to block online advertising , or the domains of known malicious resources and servers that
contain spyware , adware , and other malware . This may be achieved by adding entries for those sites to redirect requests to another address that does not
exist or to a harmless destination . Commercial software applications may be used to populate the hosts file with entries of known undesirable Internet
resources automatically . Security issues ( edit ) The hosts file may present an attack vector for malicious software . The file may be modified , for example ,
by adware , computer viruses , or trojan horse software to redirect traffic from the intended destination to sites hosting malicious or unwanted content . The
widespread computer worm Mydoom. B blocked users from visiting sites about computer security and antivirus software and also affected access from the
compromised computer to the Microsoft Windows Update website . See also ( edit ) DNSBL , a DNS - based blackhole list Content - control software Ad filtering
References ( edit ) Jump up ^ Internet Systems Consortium . `` The Most Widely Used Name Server Software : BIND '' . History of BIND . Retrieved 2017 - 07 - 01
.
</item 3>

<item 4>
hosts (file)
Network nodes typically had one address and could have many names . As local area TCP / IP computer networks gained popularity , however , the maintenance of
hosts files became a larger burden on system administrators as networks and network nodes were being added to the system with increasing frequency .
Standardization efforts , such as the format specification of the file HOSTS. TXT in RFC 952 , and distribution protocols , e.g. , the hostname server described
in RFC 953 , helped with these problems , but the centralized and monolithic nature of hosts files eventually necessitated the creation of the distributed
Domain Name System ( DNS ) . On some old systems a file named networks is present that has similar to hosts file functions containing names of networks .
Extended applications ( edit ) In its function of resolving host names , the hosts file may be used to define any hostname or domain name for use in the local
system . Redirecting local domains Some web service and intranet developers and administrators define locally defined domains in a LAN for various purposes ,
such as accessing the company 's internal resources or to test local websites in development .
</item 4>

<item 5>
hosts (file)
TXT Plan 9 / lib / ndb / hosts BeOS / boot / beos / etc / hosts Haiku / boot / common / settings / network / hosts OpenVMS UCX UCX $ HOST TCPware TCPIP $ HOST
RISC OS 3.7 , 5 ! Boot. Resources. ! Internet. files. Hosts later boot sequence ! Boot. Choices. Hardware. Disabled. Internet. Files. Hosts History ( edit ) The
ARPANET , the predecessor of the Internet , had no distributed host name database . Each network node maintained its own map of the network nodes as needed and
assigned them names that were memorable to the users of the system . There was no method for ensuring that all references to a given node in a network were
using the same name , nor was there a way to read the hosts file of another computer to automatically obtain a copy . The small size of the ARPANET kept the
administrative overhead small to maintain an accurate hosts file . Network nodes typically had one address and could have many names .
</item 5>

<item 6>
hosts (file)
Comment lines may be included ; they are indicated by an octothorpe ( # ) in the first position of such lines . Entirely blank lines in the file are ignored .
For example , a typical hosts file may contain the following : 127.0. 0.1 localhost loopback : : 1 localhost This example only contains entries for the loopback
addresses of the system and their host names , a typical default content of the hosts file . The example illustrates that an IP address may have multiple host
names ( localhost and loopback ) , and that a host name may be mapped to both IPv4 and IPv6 IP addresses , as shown on the first and second lines respectively .
Location in the file system ( edit ) The location of the hosts file in the file system hierarchy varies by operating system . It is usually named hosts ,
without an extension .
</item 6>

<item 7>
hosts (file)
5 ( Added through NetInfo or niload ) Mac OS X 10.2 and newer / etc / hosts ( a symbolic link to / private / etc / hosts ) Novell NetWare SYS : etc \ hosts OS /
2 & eComStation `` bootdrive '' : \ mptn \ etc \ Symbian Symbian OS 6.1 -- 9.0 C : \ system \ data \ hosts Symbian OS 9.1 + C : \ private \ 10000882 \ hosts
MorphOS NetStack ENVARC : sys / net / hosts AmigaOS < 4 AmiTCP : db / hosts DEVS : Internet / hosts AROS ENVARC : AROSTCP / db / hosts Android / etc / hosts ( a
symbolic link to / system / etc / hosts ) iOS iOS 2.0 and newer / etc / hosts ( a symbolic link to / private / etc / hosts ) TOPS - 20 < SYSTEM > HOSTS.
</item 7>

<item 8>
hosts (file)
Hosts ( file ) - wikipedia Hosts ( file ) Jump to : navigation , search For other uses , see Host ( disambiguation ) . The computer file hosts is an operating
system file that maps hostnames to IP addresses . It is a plain text file . Originally a file named HOSTS. TXT was manually maintained and made available via
file sharing by Stanford Research Institute for the ARPANET membership , containing the hostnames and address of hosts as contributed for inclusion by member
organizations . The Domain Name System , first described in 1983 and implemented in 1984 , automated the publication process and provided instantaneous and
dynamic hostname resolution in the rapidly growing network . In modern operating systems , the hosts file remains an alternative name resolution mechanism ,
configurable often as part of facilities such as the Name Service Switch as either the primary method or as a fallback method . Contents ( hide ) 1 Purpose 2
File content 3 Location in the file system 4 History 5 Extended applications 6 Security issues 7 See also 8 References 9 External links Purpose ( edit ) The
hosts file is one of several system facilities that assists in addressing network nodes in a computer network .
</item 8>

<item 9>
hosts (file)
It is a common part of an operating system 's Internet Protocol ( IP ) implementation , and serves the function of translating human - friendly hostnames into
numeric protocol addresses , called IP addresses , that identify and locate a host in an IP network . In some operating systems , the contents of the hosts file
is used preferentially to other name resolution methods , such as the Domain Name System ( DNS ) , but many systems implement name service switches , e.g. ,
nsswitch. conf for Linux and Unix , to provide customization . Unlike remote DNS resolvers , the hosts file is under the direct control of the local computer 's
administrator . File content ( edit ) This section needs additional citations for verification . Please help improve this article by adding citations to
reliable sources . Unsourced material may be challenged and removed . ( April 2017 ) ( Learn how and when to remove this template message ) The hosts file
contains lines of text consisting of an IP address in the first text field followed by one or more host names . Each field is separated by white space -- tabs
are often preferred for historical reasons , but spaces are also used .
</item 9>

<item 10>
hosts (file)
It is usually named hosts , without an extension . Operating System Version ( s ) Location Unix , Unix - like , POSIX / etc / hosts Microsoft Windows 3.1 %
WinDir % \ HOSTS 95 , 98 , ME % WinDir % \ hosts NT , 2000 , XP , 2003 , Vista , 2008 , 7 , 2012 , 8 , 10 % SystemRoot % \ System32 \ drivers \ etc \ hosts
Windows Mobile , Windows Phone Registry key under HKEY_LOCAL_MACHINE \ Comm \ Tcpip \ Hosts Apple Macintosh 9 and earlier Preferences or System folder Mac OS X
10.0 -- 10.1.
</item 10>
=== END ITEM id=f790408fac52ec61 ===

=== ITEM id=cbacb5782347cce6 ===
Gold sentences and facts:
Sentence 0: It was confirmed on 25 January 2018 , that Dancing on Ice had been recommissioned for an eleventh series to air in 2019 .
  Fact 0.0: On 25 January 2018, it was confirmed that Dancing on Ice had been recommissioned.
  Fact 0.1: Dancing on Ice was recommissioned for an eleventh series.
  Fact 0.2: The eleventh series is to air in 2019.

Retrieved items:
<item 1>
Dancing on Ice
AudienceTickets.net . Retrieved 24 December 2012 . Jump up ^ ( 1 ) Archived 26 January 2013 at Archive.is ^ Jump up to : Eames , Tom ( 21 May 2013 ) . `` '
Dancing on Ice ' to air final series in 2014 , confirms Torvill and Dean '' . Digital Spy . Retrieved 21 May 2013 . ^ Jump up to : `` ' Dancing on Ice ' to
survive axe ? '' . Digital Spy. 24 June 2013 . ^ Jump up to : `` Dancing on Ice to end in 2014 '' . ITV Press Centre . 22 October 2013 . Jump up ^ `` Dancing on
Ice to return in 2018 '' . ITV Press Centre . 4 September 2017 . Jump up ^ `` Dancing on Ice confirms Ashley Banjo as judge for 2018 series '' . Digital Spy .
Retrieved 19 October 2017 . Jump up ^ `` Is Jason Gardiner returning to Dancing on Ice ? Judging line - up of ITV show CONFIRMED '' . Radio Times .
</item 1>

<item 2>
Dancing on Ice
Daniel Whiston Robin Cousins Karen Barber Nicky Slater Jason Gardiner Emma Bunton Michael Ball ( week 6 ) Angela Rippon ( week 7 ) 6 9 January 2011 27 March
2011 16 Sam Attwater Brianne Delcourt Robin Cousins Jason Gardiner Emma Bunton N / A 7 8 January 2012 25 March 2012 15 Matthew Wolfenden Nina Ulanova Phillip
Schofield Christine Bleakley Robin Cousins Louie Spence Katarina Witt 8 6 January 2013 10 March 2013 12 10 Beth Tweddle Daniel Whiston Robin Cousins Karen
Barber Jason Gardiner Ashley Roberts 9 5 January 2014 9 March 2014 14 Ray Quinn Maria Filippov Nicky Slater ( weeks 6 & 7 ) 10 7 January 2018 11 March 2018 12
Jake Quickenden Vanessa Bauer Phillip Schofield Holly Willoughby Jordan Banjo ( Backstage ) Christopher Dean Jayne Torvill Jason Gardiner Ashley Banjo N / A
Presenters , judges and coaches Presenters gallery Phillip Schofield ( 2006 -- 2014 ,
</item 2>

<item 3>
Dancing on Ice
23 27 March 2011 9.35 2011 8.15 7 23 8 January 2012 8.42 25 March 2012 7.01 2012 6.63 8 19 6 January 2013 7.49 10 March 2013 7.36 2013 6.32 9 19 5 January 2014
6.76 9 March 2014 5.98 2014 5.04 10 10 7 January 2018 8.56 11 March 2018 2018 Awards Year Group Award Result Reference ( s ) 2006 National Television Awards
Most Popular Entertainment Programme Nominated 2007 National Television Awards Most Popular Talent Show Nominated 2008 National Television Awards Most Popular
Talent Show Nominated TV Quick Awards Best Talent Show Nominated 2009 National Television Awards Most Popular Talent Show Nominated TV Quick Awards Best Talent
Show Nominated RTS Craft & Design Award Best Costume Design Won RTS Craft & Design Award Best Multicamera Work Nominated RTS Craft & Design Award Best Costume
Design Nominated 2011 National Television Awards Most Popular Talent Show Nominated RTS
</item 3>

<item 4>
Dancing on Ice
These required elements included : Assisted and unassisted jumps Use of a prop Forming a spiral position Flying above the ice , suspended by a harness A one
footed spin Shadow steps Classic moves of Torvill and Dean , sometimes with a choice between them Cross rolls Toe step sequences Pair spin One unique move Solo
skate Change of edge As of Series 9 , the required skating elements format has been removed . Main series results Series 1 ( 2006 ) Main article : Dancing on
Ice ( series 1 ) The first series began on Saturday 14 January 2006 and ended on Saturday 4 March 2006 , including ten celebrity skaters . The co-presenters
were Phillip Schofield and Holly Willoughby , while The Ice Panel consisted of one - time World Junior Figure Skating champion Nicky Slater , choreographer
Karen Kresge , theatre producer Jason Gardiner , two - time NHK Trophy champion Karen Barber and 1980 Olympic champion Robin Cousins .
</item 4>

<item 5>
Dancing on Ice
As part of ITV 's new revamped schedule at the start of 2008 , from the third series , the show aired on a Sunday night and featured updated music and new
titles as well as redesigned graphics . The series 3 finale was a massive draw , pulling an average 11.7 million viewers ( up from 9.6 million the previous year
) peaking at 12.6 million viewers over the two - hour slot , up over a million from series 2 . The seventh series began on 8 January 2012 , with Torvill and
Dean as coaches for which they will be paid £ 250,000 each series . Christine Bleakley was announced as the new co-presenter of the show replacing Holly
Willoughby , whilst Katarina Witt and Louie Spence replaced Emma Bunton and Jason Gardiner on the judging panel . On 20 November 2012 , it was announced that
Gardiner was to return to the panel for the 2013 series , replacing Spence .
</item 5>

<item 6>
Dancing on Ice
Dancing on Ice - wikipedia Dancing on Ice Jump to : navigation , search This article is about the British series . For elsewhere , see Dancing on Ice around the
world . For the most recent series , see Dancing on Ice ( series 10 ) . Dancing on Ice Title -- card used for the 2018 revival Created by ITV Studios Creative
director ( s ) Christopher Dean ( 2006 -- 2014 ) Jayne Torvill ( 2006 -- 2014 ) Presented by Phillip Schofield Holly Willoughby Christine Bleakley Jordan Banjo
( backstage ) Judges Robin Cousins Karen Barber Jason Gardiner Karen Kresge Nicky Slater Natalia Bestemianova Ruthie Henshall Emma Bunton Louie Spence Katarina
Witt Ashley Roberts Christopher Dean Jayne Torvill Ashley Banjo Voices of Tony Gubba ( 2006 -- 13 ) Simon Reed ( 2013 -- 14 ) Matt Chapman ( 2018 ) Sam
Matterface ( 2018 -- ) Theme music composer Paul Farrer Country of origin United Kingdom Original language ( s ) English No. of series 10 No.
</item 6>

<item 7>
Dancing on Ice
N / A Todd Sand Kelly Holmes N / A Alexandra Schauman N / A Dr. Hilary Jones Dominic Cork Sam Nixon N / A Todd Carty N / A Tamara Sharp Andi Peters N / A Olga
Sharutenko John Barrowman N / A Keith Chegwin N / A Matej Silecky N / A Brooke Vincent Jenna Smith N / A Luke Campbell N / A Katie Stainsby N / A Vanilla Ice N
/ A Gary Lucy N / A Nina Ulanova N / A Steven Arnold Matthew Wolfenden N / A Kyran Bracken N / A Doug Webster Andrea McLean N / A Daniel Whiston Gaynor Faye
Emily Symons Linda Lusardi Roxanne Pallett Hayley Tamaddon Kerry Katona Jennifer Ellison Beth Tweddle Hayley Tamaddon Cheryl Baker Stuart Widdall N / A Coleen
Nolan Tana Ramsay N / A Michael Zenezini N / A Chloe Madeley N / A Lauren Goodger N / A Required elements From Series 1 to 8 , there was an element that the
skaters were required to perform as part of their routines .
</item 7>

<item 8>
Dancing on Ice
Dancing on Ice at itv.com Dancing on Ice at stv.tv Dancing on Ice on Facebook Dancing on Ice on Twitter Dancing on Ice on IMDb Dancing on Ice Series 5 6 7 8 9
10 Winners Celebrities Gaynor Faye Kyran Bracken Suzanne Shaw Ray Quinn Hayley Tamaddon Sam Attwater Matthew Wolfenden Beth Tweddle Jake Quickenden
Professionals Daniel Whiston Melanie Lambert Matt Evers Maria Filippov Brianne Delcourt Nina Ulanova Vanessa Bauer Other Professionals Viktoria Borzenkova
Brooke Castile Hamish Gaman Isabelle Gauthier Mark Hanretty Jodeyne Higgins Florentine Houdinière Marika Humphreys Sylvain Longchambon Pamela O'Connor Colin
Ratushniak Sean Rice Łukasz Różycki Todd Sand Olga Sharutenko Michael Zenezini Dancing on Ice around the world Versions United Kingdom United States Poland
Greece Australia Retrieved from `` https://en.wikipedia.org/w/index.
</item 8>

<item 9>
Dancing on Ice
The series will be filmed at RAF Bovingdon . Celebrity Known for Professional partner Status Candice Brown The Great British Bake Off winner Matt Evers
Eliminated 1st Stephanie Waring Actress Sylvain Longchambon Eliminated 2nd Perri Shakes - Drayton Athlete Hamish Gaman Eliminated 3rd Cheryl Baker Singer (
Bucks Fizz ) & TV presenter Daniel Whiston Eliminated 4th Lemar Singer , songwriter & producer Melody Le Moal Eliminated 5th Antony Cotton Actor Brandee Malto
Eliminated 6th & 7th Donna Air Actress & TV presenter Mark Hanretty Alex Beresford Weather forecaster Brianne Delcourt Eliminated 8th Kem Cetinay Reality TV
star Alex Murphy Eliminated 9th Max Evans Rugby union player Ale Izquierdo Third place Brooke Vincent Actress Matej Silecky Runner - up Jake Quickenden Singer &
reality TV star Vanessa Bauer Winner Series 11 ( 2019 ) It was announced on 25 January 2018 that Dancing on Ice would return for an eleventh series in 2019 .
The presenters , judges and participating celebrities are yet to be confirmed by ITV .
</item 9>

<item 10>
Dancing on Ice
The contestants for the first series were : Celebrity Known for Professional partner Status Tamara Beckwith Socialite Sergey Malyshev Eliminated 1st Andi Peters
TV presenter Tamara Sharp Eliminated 2nd Andrea McLean TV presenter Doug Webster Eliminated 3rd John Barrowman Actor & singer Olga Sharutenko Eliminated 4th
Sean Wilson Actor Marika Humphreys Eliminated 5th Kelly Holmes Olympic middle distance runner Todd Sand Eliminated 6th David Seaman Retired footballer Pam
O'Connor Eliminated 7th Bonnie Langford Actress Matt Evers Third place Stefan Booth Actor & singer Kristina Lenko Runner - up Gaynor Faye Actress Daniel Whiston
Winner Series 2 ( 2007 ) Main article : Dancing on Ice ( series 2 ) The second series began on Saturday 20 January 2007 . In it , 11 celebrities competed ,
compared to ten in the first series . The final was held on Saturday 17 March 2007 . Phillip Schofield and Holly Willoughby returned to co-present , while Nicky
Slater , Jason Gardiner , Karen Barber and Robin Cousins returned to The Ice Panel .
</item 10>
=== END ITEM id=cbacb5782347cce6 ===

=== ITEM id=e660a0e0dd1c4d05 ===
Gold sentences and facts:
Sentence 0: Top 20 rankings as of 16 October 2017 Rank Change Team Points Germany 1631 Brazil 1619 Portugal 1446 Argentina 1445 5 Belgium 1333 6 Poland 1323 7 France 1226 8 Spain 1218 9 Chile 1173 10 Peru 1160 11 Switzerland 1134 12 England 1116 13 Colombia 1095 14 Wales 1072 15 Italy 1066 16 Mexico 1060 17 Uruguay 1034 18 Croatia 1013 19 7 Denmark 1001 20 9 Netherlands 931 * Change from 14 September 2017 Complete rankings at FIFA.com
  Fact 0.0: The top 20 rankings are as of 16 October 2017.
  Fact 0.1: Germany is ranked 1 with 1631 points.
  Fact 0.2: Brazil is ranked 2 with 1619 points.
  Fact 0.3: Portugal is ranked 3 with 1446 points.
  Fact 0.4: Argentina is ranked 4 with 1445 points.
  Fact 0.5: Belgium is ranked 5 with 1333 points.
  Fact 0.6: Poland is ranked 6 with 1323 points.
  Fact 0.7: France is ranked 7 with 1226 points.
  Fact 0.8: Spain is ranked 8 with 1218 points.
  Fact 0.9: Chile is ranked 9 with 1173 points.
  Fact 0.10: Peru is ranked 10 with 1160 points.
  Fact 0.11: Switzerland is ranked 11 with 1134 points.
  Fact 0.12: England is ranked 12 with 1116 points.
  Fact 0.13: Colombia is ranked 13 with 1095 points.
  Fact 0.14: Wales is ranked 14 with 1072 points.
  Fact 0.15: Italy is ranked 15 with 1066 points.
  Fact 0.16: Mexico is ranked 16 with 1060 points.
  Fact 0.17: Uruguay is ranked 17 with 1034 points.
  Fact 0.18: Croatia is ranked 18 with 1013 points.
  Fact 0.19: Denmark is ranked 19 with 1001 points and has a rank change of 7.
  Fact 0.20: Netherlands is ranked 20 with 931 points and has a rank change of 9.
  Fact 0.21: The rank changes are measured from 14 September 2017.
  Fact 0.22: The complete rankings are available at FIFA.com.

Retrieved items:
<item 1>
FIFA World Rankings
Archived from the original on 25 December 2011 . Retrieved 13 April 2011 . Jump up ^ `` Olympics -- updated draw procedure , with teams '' . CAFOnline.com .
Confederation of African Football . 12 April 2011 . Retrieved 13 April 2011 . ^ Jump up to : `` The world rankings riddle '' . BBC Sport. 21 December 2000 .
Jump up ^ `` FIFA adapting new world rankings '' ( PDF ) . Associated Press . 2 June 2006 . Retrieved 23 July 2017 . Jump up ^ `` FIFA chief praises IFA 's work
'' . The Jerusalem Post . 29 August 2008 . Archived from the original on 27 April 2011 . Retrieved 13 February 2009 . Blatter was also asked about Israel 's
unrealistic FIFA ranking . Despite failing to qualify for a major tournament in 38 years , the national team is currently ranked number 16 in the world , ahead
of such team as Greece , Sweden and Denmark , and just two places below England .
</item 1>

<item 2>
FIFA World Rankings
The change is perceived to respond to criticisms that the FIFA World Rankings , based upon the previous calculation method in use from January 1999 to June 2006
, did not effectively reflect the relative strengths of the national teams . Alternative systems have been devised , such as the World Football Elo Ratings ,
based on the Elo rating system used in chess and Go , ranking teams . In September 2017 , FIFA announced they were reviewing the ranking system and will decide
after the end of the 2018 FIFA World Cup qualification if any changes are to be made to improve the ranking .
</item 2>

<item 3>
FIFA World Rankings
From the following August , this list was more frequently updated , to be published most months . Significant changes were implemented in January 1999 and again
in July 2006 , as a reaction to criticisms of the system . Historical records of the rankings , such as listed at FIFA.com , reflect the method of calculation
in use at the time , as the current method has not been applied retrospectively to rankings prior to July 2006 . Membership of FIFA has expanded from 167 to 209
since the rankings began ; all 209 members are currently included in the rankings . 1993 -- 1998 calculation method ( edit ) The ranking formula used from
August 1993 until December 1998 was very simplistic and quickly became noticed for its lack of supporting factors . When the rankings were initially introduced
, a team received one point for a draw or three for a victory in FIFA - recognised matches -- much the same as a traditional league scoring system . This was a
quite simplistic approach , however , and FIFA quickly realised that there were many factors affecting international matches .
</item 3>

<item 4>
FIFA World Rankings
Jump up ^ `` Soccer : Quirky FIFA rankings system sees Israel climb to # 15 '' . The Jerusalem Post . 13 November 2008 . Archived from the original on 16
November 2008 . Retrieved 13 February 2009 . Israel continued its baffling rise up the FIFA rankings on Wednesday , climbing one more position to a best - ever
15th in the world . Amazingly , the national team , which has n't reached a major tournament since the 1970 World Cup , is ahead of such teams as Greece ( 18 )
, Nigeria ( 22 ) , Sweden ( 29 ) , Scotland ( 33 ) , Denmark ( 34 ) and the Republic of Ireland ( 36 ) Jump up ^ `` The Last Word : It 's time to get rid of the
FIFA rankings '' . The Jerusalem Post . 14 November 2008 . Archived from the original on 16 November 2008 . Retrieved 13 February 2009 .
</item 4>

<item 5>
FIFA World Rankings
In July 2014 , Germany took over the lead once again , having won the 2014 FIFA World Cup . In July 2015 , Argentina reached the top spot for the first time
since 2008 , after reaching both the 2014 FIFA World Cup Final , as well as the 2015 Copa America Final . In November 2015 , Belgium became the leader in the
FIFA rankings for the first time , after topping their Euro 2016 qualifying group . Belgium led the rankings until April 2016 , when Argentina returned to the
top . On April 6 , 2017 , Brazil returned to the No. 1 spot for the first time since just prior to the 2010 World Cup , but Germany regained the top spot in
July after winning the Confederations Cup . Uses of the rankings ( edit ) The rankings are used by FIFA to rank the progression and current ability of the
national football teams of its member nations , and claims that they create `` a reliable measure for comparing national A-teams '' . They are used as part of
the calculation , or the entire grounds to seed competitions .
</item 5>

<item 6>
FIFA World Rankings
FIFA World rankings - wikipedia FIFA World rankings This is the latest accepted revision , reviewed on 25 October 2017 . Jump to : navigation , search This
article is about the men 's rankings . For the women 's rankings , see FIFA Women 's World Rankings . Top 20 rankings as of 16 October 2017 Rank Change Team
Points Germany 1631 Brazil 1619 Portugal 1446 Argentina 1445 5 Belgium 1333 6 Poland 1323 7 France 1226 8 Spain 1218 9 Chile 1173 10 Peru 1160 11 Switzerland
1134 12 England 1116 13 Colombia 1095 14 Wales 1072 15 Italy 1066 16 Mexico 1060 17 Uruguay 1034 18 Croatia 1013 19 7 Denmark 1001 20 9 Netherlands 931 * Change
from 14 September 2017 Complete rankings at FIFA.com The FIFA World Ranking is a ranking system for men 's national teams in association football , currently
led by Germany .
</item 6>

<item 7>
FIFA World Rankings
Travour.com. 1 June 2006 . Archived from the original on 14 June 2006 . Jump up ^ Cummings , Michael . `` FIFA World Rankings Place Brazil 18th , Reinforce
Flawed Nature of System '' . Retrieved 21 May 2013 . Jump up ^ Freitas , Paulo . `` Scolari blames FIFA for Brazil 's poor ranking '' . Retrieved 16 June 2013 .
^ Jump up to : Burnton , Simon ( 17 October 2013 ) . `` With a bit of planning England could have been seeded ahead of the Swiss '' . The Guardian . Retrieved
15 June 2015 . Jump up ^ `` Romania , Wales set to be surprise World Cup top seeds '' . USA Today . Associated Press . 23 July 2015 . Retrieved 29 July 2015 .
Romania took advice from a consultant who decoded some mysteries of a ranking system used by FIFA since 1992 . Jump up ^ Gavril , Victor ( 5 July 2015 ) .
</item 7>

<item 8>
FIFA World Rankings
Retrieved 13 February 2009 . So the news this week that Israel has risen another place to number 15 in the latest FIFA rankings , above Mexico ( 25 ) , Nigeria
( 22 ) , the US ( 24 ) and Colombia ( 40 ) , only serves to underline the futility of the ranking system . Jump up ^ `` FIFA Ranking : November 2008 preview (
II ) '' . Football-Rankings.info. 20 October 2008 . Retrieved 13 February 2009 . Israel will improve ( again ! ) their best ever ranking , climbing one place to
15th . With a win ( instead of a draw ) against Latvia they would have been tied for 8th place with Russia . Jump up ^ Alistair Tweedale ( 13 October 2015 ) .
`` Belgium rise to No1 in Fifa world rankings after they beat Israel -- despite playing one tournament in 13 years '' . The Telegraph . Retrieved 14 November
2015 . Jump up ^ `` FIFA Rankings '' . Travour.com. 1 June 2006 .
</item 8>

<item 9>
FIFA World Rankings
The teams of the member nations of FIFA , football 's world governing body , are ranked based on their game results with the most successful teams being ranked
highest . The rankings were introduced in December 1992 , and eight teams ( Argentina , Belgium , Brazil , France , Germany , Italy , the Netherlands and Spain
) have held the top position , of which Brazil have spent longest ranked first . A points system is used , with points being awarded based on the results of all
FIFA - recognised full international matches . Under the existing system , rankings are based on a team 's performance over the last four years , with more
recent results and more significant matches being more heavily weighted to help reflect the current competitive state of a team . The ranking system was most
recently revamped after the 2006 World Cup , with the first edition of the new series of rankings issued on 12 July 2006 . The most significant change is that
the rankings are now based on results over the previous four years instead of the previous eight years .
</item 9>

<item 10>
FIFA World Rankings
FIFA. 24 August 2011 . Archived from the original on 25 September 2011 . Retrieved 26 August 2011 . Jump up ^ Team of the Year Award 2010 Archived 18 December
2010 at the Wayback Machine . on the FIFA website Jump up ^ Best Mover of the Year Archived 20 July 2008 at the Wayback Machine . on the FIFA website Jump up ^
Spain on top and Wales highest climber on the FIFA website Jump up ^ Top Team and the Best Mover of the Year on the FIFA website ( PDF ) Jump up ^ Spain finish
2012 on top , Colombia in fifth . FIFA.com . Retrieved on 12 August 2013 . Jump up ^ `` Spain on top , Ukraine highest climber '' . fifa.com. 19 December 2013 .
Retrieved 19 December 2013 . Jump up ^ `` Germany conquer 2014 , Belgium , Slovakia impress '' . fifa.com. 18 December 2014 . Retrieved 18 December 2014 . Jump
up ^ `` Belgium and Turkey claim awards , Hungary return '' .
</item 10>
=== END ITEM id=e660a0e0dd1c4d05 ===
