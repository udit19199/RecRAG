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

=== ITEM id=e3c0fc098e86f349 ===
Gold sentences and facts:
Sentence 0: The provisions relating to the right to property were changed a number of times . The Forty - Fourth Amendment of 1978 deleted the right to property from the list of fundamental rights A new provision , Article 300 - A , was added to the constitution , which provided that `` no person shall be deprived of his property save by authority of law '' . Thus if a legislature makes a law depriving a person of his property , there would be no obligation on the part of the State to pay anything as compensation . The aggrieved person shall have no right to move the court under Article 32 . Thus , the right to property is no longer a fundamental right , though it is still a constitutional right . If the government appears to have acted unfairly , the action can be challenged in a court of law by aggrieved citizens .
  Fact 0.0: The provisions relating to the right to property were changed a number of times.
  Fact 0.1: The Forty-Fourth Amendment of 1978 deleted the right to property from the list of fundamental rights.
  Fact 0.2: A new provision, Article 300-A, was added to the constitution.
  Fact 0.3: Article 300-A provides that no person shall be deprived of his property save by authority of law.
  Fact 0.4: If a legislature makes a law depriving a person of his property, the State has no obligation to pay anything as compensation.
  Fact 0.5: An aggrieved person shall have no right to move the court under Article 32 in that case.
  Fact 0.6: The right to property is no longer a fundamental right.
  Fact 0.7: The right to property is still a constitutional right.
  Fact 0.8: If the government appears to have acted unfairly, the action can be challenged in a court of law by aggrieved citizens.

Retrieved items:
<item 1>
Fundamental rights in India
The right to constitutional remedies is present for enforcement of Fundamental Rights . The right to privacy is an intrinsic part of Article 21 that protects
life and liberty of the citizens . Fundamental rights for Indians have also been aimed at overturning the inequalities of pre-independence social practices .
Specifically , they have also been used to abolish untouchability and thus prohibit discrimination on the grounds of religion , race , caste , sex , or place of
birth . They also forbid trafficking of human beings and forced labour ( a crime ) . They also protect cultural and educational rights of religious and
linguistic minorities by allowing them to preserve their languages and also establish and administer their own education institutions . They are covered in Part
III ( Articles 12 to 35 ) of Indian constitution .
</item 1>

<item 2>
Fundamental rights in India
. The Times of India . Retrieved 8 July 2010 . Jump up ^ `` Court rejects plea to make property a fundamental right '' . The Hindu . Chennai , India . 19
October 2010 . Jump up ^ `` Education is now a fundamental right of every child '' . Hindustan Times . 1 April 2010 . Archived from the original on 9 July 2010
. Retrieved 8 July 2010 . Footnotes ( edit ) Links to related articles ( hide ) Government of India Constitution Constitution Fundamental Rights , Directive
Principles and Fundamental Duties Federalism ( Union List State List Concurrent List ) Executive Union Government Prime Minister ( list ) Cabinet Ministers
President ( list ) Vice President ( list ) Civil Services All India Services ( IAS IFS / IFoS IPS ) Cabinet Secretary Secretaries ( Defence Finance Foreign Home
) State governments Chief Minister ( list ) Chief Secretary Governors ( list ) Legislative Parliament Lok Sabha ( Speaker ) Rajya Sabha ( Chairman ) Vidhan
Sabha Vidhan Parishad Leader of the Opposition Judiciary Supreme Court ( Chief Justice ) High Courts District Courts Retrieved from ``
https://en.wikipedia.org/w/index.
</item 2>

<item 3>
Fundamental rights in India
The provisions relating to the right to property were changed a number of times . The Forty - Fourth Amendment of 1978 deleted the right to property from the
list of fundamental rights A new provision , Article 300 - A , was added to the constitution , which provided that `` no person shall be deprived of his
property save by authority of law '' . Thus if a legislature makes a law depriving a person of his property , there would be no obligation on the part of the
State to pay anything as compensation . The aggrieved person shall have no right to move the court under Article 32 . Thus , the right to property is no longer
a fundamental right , though it is still a constitutional right . If the government appears to have acted unfairly , the action can be challenged in a court of
law by aggrieved citizens . The liberalisation of the economy and the government 's initiative to set up special economic zones has led to many protests by
farmers and have led to calls for the reinstatement of the fundamental right to private property .
</item 3>

<item 4>
Fundamental rights in India
They are covered in Part III ( Articles 12 to 35 ) of Indian constitution . Contents ( hide ) 1 Genesis 2 Significance and characteristics 3 Right to equality 4
Right to freedom 5 Right against exploitation 6 Right to freedom of religion 7 Right to life 8 Cultural and Educational rights 9 Right to constitutional
remedies 10 Right to Privacy 11 Critical analysis 12 Amendments 13 Fundamental rights not sacrosanct 13.1 Validity of Article 31B 13.2 Amendment to Article 31C
13.3 Right to property 13.4 Right to education 14 See also 15 References 16 Footnotes Genesis ( edit ) The development of constitutionally guaranteed
fundamental human rights in India was inspired by historical examples such as England 's Bill of Rights ( 1689 ) , the United States Bill of Rights ( approved
on 17 September 1787 , final ratification on 15 December 1791 ) and France 's Declaration of the Rights of Man ( created during the revolution of 1789 , and
ratified on 26 August 1789 ) .
</item 4>

<item 5>
Fundamental rights in India
These fundamental rights help not only in protection but also the prevention of gross violations of human rights . They emphasise on the fundamental unity of
India by guaranteeing to all citizens the access and use of the same facilities , irrespective of background . Some fundamental rights apply for persons of any
nationality whereas others are available only to the citizens of India . The right to life and personal liberty is available to all people and so is the right
to freedom of religion . On the other hand , freedoms of speech and expression and freedom to reside and settle in any part of the country are reserved to
citizens alone , including non-resident Indian citizens . The right to equality in matters of public employment can not be conferred to overseas citizens of
India . Fundamental rights primarily protect individuals from any arbitrary state actions , but some rights are enforceable against individuals . For instance ,
the Constitution abolishes untouchability and also prohibits begar . These provisions act as a check both on state action as well as the action of private
individuals . However , these rights are not absolute or uncontrolled and are subject to reasonable restrictions as necessary for the protection of general
welfare . They can also be selectively curtailed .
</item 5>

<item 6>
Fundamental rights in India
Right to Privacy ( edit ) The right to privacy is protected as an intrinsic part of the right to life and personal liberty under Article 21 and as a part of the
freedoms guaranteed by Part III of the Constitution . It protects the inner sphere of the individual from interference from both State , and non-State actors
and allows the individuals to make autonomous life choices . On 24 August 2017 the Supreme Court ruled that : `` Right to Privacy is an integral part of Right
to Life and Personal Liberty guaranteed in Article 21 of the Constitution , '' -- Supreme Court Critical analysis ( edit ) The fundamental rights have been
revised for many reasons . Political groups have demanded that the right to work , the right to economic assistance in case of unemployment , old age , and
similar rights be enshrined as constitutional guarantees to address issues of poverty and economic insecurity , though these provisions have been enshrined in
the Directive Principles of state policy . The right to freedom and personal liberty has a number of limiting clauses , and thus have been criticised for
failing to check the sanctioning of powers often deemed `` excessive '' . There is also the provision of preventive detention and suspension of fundamental
rights in times of Emergency .
</item 6>

<item 7>
Fundamental rights in India
The six fundamental rights recognised by the Indian constitution are the right to equality , right to freedom , right against exploitation , right to freedom of
religion , cultural and educational rights , right to constitutional remedies and right to privacy . The right to equality includes equality before law ,
prohibition of discrimination on grounds of religion , race , caste , gender or place of birth , and equality of opportunity in matters of employment ,
abolition of untouchability and abolition of titles . The right to freedom includes freedom of speech and expression , assembly , association or union or
cooperatives , movement , residence , and right to practice any profession or occupation , right to life and liberty , protection in respect to conviction in
offences and protection against arrest and detention in certain cases . The right against exploitation prohibits all forms of forced labour , child labour and
trafficking of human beings . The right to freedom of religion includes freedom of conscience and free profession , practice , and propagation of religion ,
freedom to manage religious affairs , freedom from certain taxes and freedom from religious instructions in certain educational institutes . Cultural and
educational rights preserve the right of any section of citizens to conserve their culture , language or script , and right of minorities to establish and
administer educational institutions of their choice .
</item 7>

<item 8>
Fundamental rights in India
Article 1 of Part 1 of the Indian constitution , defines India ( Bharat ) as union of states . In nutshell , India is its people not its land as enshrined in
its constitution . Since speedy trial is not the constitutional right of the citizens , the cases involving violations of fundamental rights take inordinate
time for resolution by the Supreme Court which is against the legal maxim ' justice delayed is justice denied ' . Amendments ( edit ) Changes to the fundamental
rights require a constitutional amendment , which has to be passed by a special majority of both houses of Parliament . This means that an amendment requires
the approval of two - thirds of the members present and voting . However , the number of members voting in support of the amendment shall not be less than the
simple majority of the total members of a house -- whether the Lok Sabha or Rajya Sabha . Fundamental rights not sacrosanct ( edit ) While deciding the
Golaknath case in February 1967 , Supreme Court ruled that the Parliament has no power to curtail the fundamental rights .
</item 8>

<item 9>
Fundamental rights in India
In 2007 , Supreme Court ruled that there could not be any blanket immunity from judicial review for the laws inserted in the Ninth Schedule . Apex court also
stated it shall examine laws included in the Ninth Schedule after 1973 for any incompatibility with the basic structure doctrine . Amendment to Article 31C (
edit ) Section 4 of the 42nd Amendment , had changed Article 31C of the constitution to accord precedence to the Directive Principles ( earlier applicable only
to clauses b & c of Article 39 ) over the fundamental rights of individuals . In Minerva Mills v. Union of India case , supreme court ruled that the amendment
to the Article 31C is not valid and ultra vires . Right to property ( edit ) The Constitution originally provided for the right to property under Articles 19
and 31 . Article 19 guaranteed to all citizens the right to acquire , hold and dispose of property . Article 31 provided that `` no person shall be deprived of
his property save by authority of law . '' It also provided that compensation would be paid to a person whose property has been taken for public purposes .
</item 9>

<item 10>
Fundamental rights in India
Jump up ^ Sovereign means possessing supreme or ultimate power . `` in modern democracies the people 's will is in theory sovereign '' synonyms : supreme ,
absolute , unlimited , unrestricted , unrestrained , unbounded , boundless , infinite , ultimate , total , unconditional , full , utter , paramount ; principal
, chief , dominant , predominant Jump up ^ `` What causes judicial delay ? Judgements diluting time frames in Code of Civil Procedure worsen the problem of
adjournments '' . Retrieved 13 September 2017 . Jump up ^ `` Paras 316 and 317 of Kesavananda Bharati v. State of Kerala , ( AIR 1973 SC 1461 ) '' . 1973 .
Retrieved 21 October 2016 . Jump up ^ `` IX Schedule laws open to review '' . 2007 . Retrieved 21 October 2017 . Jump up ^ 44th Amendment Act , 1978 . Jump up ^
Mahapatra , Dhananjay ( 28 February 2009 ) . `` Should right to property return ? '' . The Times of India . Retrieved 8 July 2010 .
</item 10>
=== END ITEM id=e3c0fc098e86f349 ===

=== ITEM id=f0460241675b6ec7 ===
Gold sentences and facts:
Sentence 0: The Battle of Lake Erie , sometimes called the Battle of Put - in - Bay , was fought on 10 September 1813 , in Lake Erie off the coast of Ohio during the War of 1812 . Nine vessels of the United States Navy defeated and captured six vessels of the British Royal Navy . This ensured American control of the lake for the rest of the war , which in turn allowed the Americans to recover Detroit and win the Battle of the Thames to break the Indian confederation of Tecumseh . It was one of the biggest naval battles of the War of 1812 .
  Fact 0.0: The Battle of Lake Erie is sometimes called the Battle of Put-in-Bay.
  Fact 0.1: The Battle of Lake Erie was fought on 10 September 1813.
  Fact 0.2: The Battle of Lake Erie was fought in Lake Erie off the coast of Ohio.
  Fact 0.3: The Battle of Lake Erie was fought during the War of 1812.
  Fact 0.4: Nine vessels of the United States Navy defeated six vessels of the British Royal Navy in the Battle of Lake Erie.
  Fact 0.5: Nine vessels of the United States Navy captured six vessels of the British Royal Navy in the Battle of Lake Erie.
  Fact 0.6: The Battle of Lake Erie ensured American control of the lake for the rest of the War of 1812.
  Fact 0.7: The Battle of Lake Erie allowed the Americans to recover Detroit.
  Fact 0.8: The Battle of Lake Erie allowed the Americans to win the Battle of the Thames.
  Fact 0.9: Winning the Battle of the Thames broke the Indian confederation of Tecumseh.
  Fact 0.10: The Battle of Lake Erie was one of the biggest naval battles of the War of 1812.

Retrieved items:
<item 1>
Battle of Lake Erie
Battle of Lake Erie - wikipedia Battle of Lake Erie Jump to : navigation , search Battle of Lake Erie Part of the War of 1812 Battle of Lake Erie by William
Henry Powell , painted 1865 , shows Oliver Hazard Perry transferring from Lawrence to Niagara Date 10 September 1813 Location Lake Erie , near Put - in - Bay ,
Ohio Result Decisive American victory Belligerents United States United Kingdom Commanders and leaders Oliver Hazard Perry Jesse Elliot Robert Heriot Barclay
Robert Finnis Strength 5 schooners 3 brigs 1 sloop 2 ships 1 brig 2 schooners 1 sloop Casualties and losses 27 killed 96 wounded 1 brig severely damaged 41
killed 93 wounded 306 captured entire squadron captured Detroit frontier Tippecanoe Fort Mackinac ( 1812 ) Brownstown Maguaga Fort Dearborn Detroit Fort
Harrison Fort Wayne Wild Cat Creek Mississinewa Frenchtown Africa Point Fort Meigs Fort Stephenson Lake Erie Thames Longwoods Prairie du Chien Campbell Island
Mackinac Island ( 1814 ) Lake Huron Malcolm 's Mills Naval campaigns of the War of 1812 Atlantic Ocean USS Essex vs HMS Alert USS Constitution vs HMS
</item 1>

<item 2>
Battle of Lake Erie
Naval campaigns of the War of 1812 Atlantic Ocean USS Essex vs HMS Alert USS Constitution vs HMS Guerriere Capture of HMS Frolic USS United States vs HMS
Macedonian USS Constitution vs HMS Java Sinking of HMS Peacock Rappahannock River Capture of USS Chesapeake Capture of HMS Dominica Capture of USS Argus Capture
of HMS Boxer Capture of USS Frolic Capture of HMS Epervier Sinking of HMS Reindeer Sinking of HMS Avon Fayal Capture of USS President Capture of HMS Cyane and
HMS Levant Capture of HMS Penguin Capture of East India Company ship Nautilus East Coast Chesapeake Bay Alexandria Baltimore Hampden Fort Peter Great Lakes /
Saint Lawrence River Lake Ontario 1st Sacket 's Harbor York Fort George 2nd Sacket 's Harbor Lake Erie Fort Oswego Lake Huron Lake Champlain West Indies / Gulf
Coast La Guaira 1st Fort Bowyer Action of 13 December 1814 Lake Borgne New Orleans Fort St.
</item 2>

<item 3>
Battle of Lake Erie
Fort Bowyer Action of 13 December 1814 Lake Borgne New Orleans Fort St. Philip 2nd Fort Bowyer Pacific Ocean James Island Charles Island Nuku Hiva Downes
Expedition Porter Expedition Typee Valley Valparaiso ( Capture of USS Essex ) Seringapatam Mutiny Action of 9 May 1814 The Battle of Lake Erie , sometimes
called the Battle of Put - in - Bay , was fought on 10 September 1813 , in Lake Erie off the coast of Ohio during the War of 1812 . Nine vessels of the United
States Navy defeated and captured six vessels of the British Royal Navy . This ensured American control of the lake for the rest of the war , which in turn
allowed the Americans to recover Detroit and win the Battle of the Thames to break the Indian confederation of Tecumseh . It was one of the biggest naval
battles of the War of 1812 .
</item 3>

<item 4>
Battle of Lake Erie
Michaels Raid on Havre de Grace Massachusetts Battle of Hampden New York Battle of Big Sandy Creek Battle of Buffalo Battle of Ogdensburg Battle of Plattsburgh
Capture of Fort Niagara Raid on Black Rock Second Battle of Sacket 's Harbor Ohio Battle of Fort Stephenson Copus massacre Siege of Fort Meigs Virginia Battle
of Craney Island Raid on Alexandria Skirmish at Farnham Church Battle of Rappahannock River U.S. territories Alabama Battles of Fort Bowyer Illinois Battle of
Fort Dearborn Battle of Rock Island Rapids Siege of Prairie du Chien Indiana Battle of the Mississinewa Battle of Tippecanoe Battle of Wild Cat Creek Siege of
Fort Harrison Siege of Fort Wayne Michigan Battle of Brownstown Battle of Frenchtown Battle of Mackinac Island Battle of Maguaga Siege of Detroit Siege of Fort
Mackinac Mississippi Battle of Burnt Corn Battle of Callabee Creek Canoe Fight Battle of Holy Ground Battle of Horseshoe Bend Battle of Talladega Battle of
Tallushatchee Battles of Emuckfaw and Enotachopo Creek Fort Mims massacre Kimbell -- James Massacre Missouri Battle of Credit Island Battle of the Sink Hole
British North America Lower Canada Battle of
</item 4>

<item 5>
Battle of Lake Erie
It was one of the biggest naval battles of the War of 1812 . Contents ( hide ) 1 Background 1.1 1812 1.2 1813 1.3 Blockades of Presque Isle and Amherstburg 2
Battle 3 Casualties 4 Aftermath 5 Reconstructions and memorials 6 Reasons for the American victory 7 Order of battle 8 Footnotes 9 Notes 10 References 11
Further reading 12 External links Background ( edit ) 1812 ( edit ) When the war broke out , the British immediately seized control of Lake Erie . They already
had a small force of warships there : the sloop - of - war Queen Charlotte and the brig General Hunter . The schooner Lady Prevost was under construction and
was put into service a few weeks after the outbreak of war . These vessels were controlled by the Provincial Marine , which was a military transport service and
not a naval service . Nevertheless , the Americans lacked any counter to the British armed vessels .
</item 5>

<item 6>
Battle of Lake Erie
The Americans lost 27 killed and 96 wounded , of whom 2 later died . Of the vessels involved , the three most battered ( the American brig Lawrence and the
British ships Detroit and Queen Charlotte ) were converted into hospital ships . A gale swept the lake on 13 September and dismasted Detroit and Queen Charlotte
, further shattering the already battered ships . Once the wounded had been ferried to Erie , Lawrence was restored to service for 1814 , but the two British
ships were effectively reduced to hulks . Aftermath ( edit ) This section needs additional citations for verification . Please help improve this article by
adding citations to reliable sources . Unsourced material may be challenged and removed . ( September 2017 ) ( Learn how and when to remove this template
message ) Perry 's vessels and prizes were anchored and hasty repairs were underway near West Sister Island when Perry composed his now famous message to
Harrison . Scrawled in pencil on the back of an old envelope , Perry wrote : Dear General : We have met the enemy and they are ours . Two ships , two brigs ,
one schooner and one sloop .
</item 6>

<item 7>
Battle of Lake Erie
Nevertheless , the Americans lacked any counter to the British armed vessels . The only American warship on Lake Erie , the brig Adams , was not ready for
service at the start of the war , and when the American army of Brigadier General William Hull abandoned its invasion of Canada , Adams was pinned down in
Detroit by the British batteries at Sandwich on the Canadian side of the Detroit River . The British Major - General Isaac Brock used his control of the lake to
defeat Hull 's army at the Siege of Detroit , by cutting the American supply lines and rapidly transferring himself and some reinforcements to Amherstburg from
where they launched a successful landing on the American side of the Detroit River . The British took Adams when Detroit was surrendered , renaming her Detroit
. Together with the brig Caledonia , which had been commandeered from the Canadian North West Company , she was boarded and captured near Fort Erie on 9 October
, by American sailors and soldiers led by Lieutenant Jesse Elliot . Detroit ran aground on an island in the middle of the Niagara River and was set on fire to
prevent her being recaptured . Caledonia was taken to the navy yard at Black Rock and commissioned into the United States Navy .
</item 7>

<item 8>
Battle of Lake Erie
James Massacre Missouri Battle of Credit Island Battle of the Sink Hole British North America Lower Canada Battle of the Chateauguay First Battle of Lacolle
Mills Second Battle of Lacolle Mills Upper Canada Battle of Beaver Dams Battle of Chippawa Battle of Cook 's Mills Battle of Crysler 's Farm Battle of Fort
George Battle of Frenchman 's Creek Battle of Longwoods Battle of Lundy 's Lane Battle of Malcolm 's Mills Battle of Queenston Heights Battle of Stoney Creek
Battle of the Thames Battle of York Capture of Fort Erie Raid on Elizabethtown Raid on Port Dover Raid on Gananoque Siege of Fort Erie Spanish Empire Spanish
Florida Battle of Pensacola Naval battles Atlantic Ocean Capture of HMS Boxer Capture of HMS Cyane Capture of HMS Epervier Capture of HMS Frolic Capture of HMS
Penguin Capture of HMS Dominica Capture of USS Argus Capture of USS Chesapeake Capture of USS President Chesapeake Bay Flotilla USS Constitution vs HMS Java
Sinking of HMS Avon Battle of Fayal Sinking of HMS Peacock Sinking of HMS Reindeer USS Constitution vs HMS Guerriere USS United States vs HMS Macedonian
Caribbean Sea Battle of La Guaira Great Lakes Battle of Lake Erie
</item 8>

<item 9>
Battle of Lake Erie
The weather was hazy , and he missed them . During July and August , Barclay received two small vessels , the schooner Chippeway and the sloop Little Belt ,
which had been reconstructed at Chatham on the Thames River and attempted to complete the ship - rigged corvette HMS Detroit at Amherstburg . Because the
Americans controlled Lake Ontario and occupied the Niagara Peninsula in early 1813 , supplies for Barclay had to be carried overland from York . The American
victory earlier in the year at the Battle of York resulted in the guns ( 24 - pounder carronades ) intended for Detroit falling into American hands . Detroit
had to be completed with a miscellany of guns from the fortifications of Amherstburg . Barclay claimed at his court martial that these guns lacked flintlock
firing mechanisms and matches , and that they could be fired only by snapping flintlock pistols over powder piled in the vent holes . Barclay repeatedly
requested men and supplies from Commodore James Lucas Yeo , commanding on Lake Ontario , but received very little .
</item 9>

<item 10>
Battle of Lake Erie
USS United States vs HMS Macedonian Caribbean Sea Battle of La Guaira Great Lakes Battle of Lake Erie Battle of Fort Oswego Engagements on Lake Huron
Engagements on Lake Ontario First Battle of Sacket 's Harbor Gulf Coast Action of 13 December 1814 Battle of Lake Borgne Pacific Ocean Action off James Island
Action off Charles Island Nuku Hiva Campaign Battle of Valparaiso ( Capture of USS Essex ) See also : American Indian Wars , Creek War , Napoleonic Wars , and
Tecumseh 's War Portal definition textbooks quotes source texts media news stories Coordinates : 41 ° 39 ′ 44 '' N 82 ° 49 ′ 30 '' W ﻿ / ﻿ 41.66222 ° N 82.82500
° W ﻿ / 41.66222 ; - 82.82500 Retrieved from `` https://en.wikipedia.org/w/index.
</item 10>
=== END ITEM id=f0460241675b6ec7 ===

=== ITEM id=3db1bb3a1a3697d3 ===
Gold sentences and facts:
Sentence 0: From September 1947 , Toyota 's small - sized vehicles were sold under the name `` Toyopet '' ( トヨペット ) . The first vehicle sold under this name was the Toyopet SA , but it also included vehicles such as the Toyopet SB light truck , Toyopet Stout light truck , Toyopet Crown , Toyopet Master , and the Toyopet Corona . The word `` Toyopet ( Japanese article ) '' was a nickname given to the Toyota SA due to its small size , as the result of a naming contest the Toyota Company organized in 1947 . However , when Toyota eventually entered the American market in 1957 with the Crown , the name was not well received due to connotations of toys and pets . The name was soon dropped for the American market , but continued in other markets until the mid-1960s .
  Fact 0.0: From September 1947, Toyota's small-sized vehicles were sold under the name 'Toyopet' (トヨペット).
  Fact 0.1: The first vehicle sold under the name Toyopet was the Toyopet SA.
  Fact 0.2: Toyopet vehicles also included the Toyopet SB light truck.
  Fact 0.3: Toyopet vehicles also included the Toyopet Stout light truck.
  Fact 0.4: Toyopet vehicles also included the Toyopet Crown.
  Fact 0.5: Toyopet vehicles also included the Toyopet Master.
  Fact 0.6: Toyopet vehicles also included the Toyopet Corona.
  Fact 0.7: The word 'Toyopet' was a nickname given to the Toyota SA due to its small size.
  Fact 0.8: The Toyopet nickname resulted from a naming contest the Toyota Company organized in 1947.
  Fact 0.9: When Toyota entered the American market in 1957 with the Crown, the Toyopet name was not well received due to connotations of toys and pets.
  Fact 0.10: The Toyopet name was soon dropped for the American market.
  Fact 0.11: The Toyopet name continued in other markets until the mid-1960s.

Retrieved items:
<item 1>
Toyota
Inside Investor . Retrieved July 24 , 2013 . Jump up ^ `` The Story of Sakichi Toyoda '' . Toyota Industries Corporation . Retrieved September 22 , 2017 . Jump
up ^ `` History Of Toyota '' . Toyota . Retrieved August 15 , 2011 . Jump up ^ `` Toyota Company History from 1867 to 1939 '' . Toyota . Archived from the
original on May 10 , 2013 . Retrieved September 11 , 2010 . Jump up ^ `` Information from a sign at the Toyota Museum in Nagakute - cho , Aichi - gun , Aichi
Pref '' . Toyota . Archived from the original on November 20 , 2012 . Retrieved December 7 , 2012 . Jump up ^ Schmitt , Bertel ( July 5 , 2011 ) . `` At Home
With The Toyodas -- And A Mysterious Cherry Tree '' . Thetruthaboutcars . Retrieved August 15 , 2011 . Jump up ^ Davis , Pedr ( 1999 ) .
</item 1>

<item 2>
Toyota
Akio Toyoda became the new president and CEO of the company on June 23 , 2009 , by replacing Katsuaki Watanabe , who became the new vice chairman by replacing
Katsuhiro Nakagawa . Worldwide presence ( edit ) The Toyota Camry is assembled in several facilities around the world including Australia , China , Taiwan ,
Japan , Malaysia , the Philippines , Pakistan , Russia , Thailand , India , Vietnam , and the United States Toyota has factories in most parts of the world ,
manufacturing or assembling vehicles for local markets in Japan , Australia , India , Sri Lanka , Canada , Indonesia , Poland , South Africa , Turkey , Colombia
, the United Kingdom , the United States , France , Brazil , Portugal , and more recently , Argentina , Czech Republic , Mexico , Malaysia , Thailand , Pakistan
, Egypt , China , Vietnam , Venezuela , the Philippines , and Russia .
</item 2>

<item 3>
Toyota
Toyota Technical Center , Ann Arbor Charter Township , Michigan Toyota has a large presence in the United States with six major assembly plants in Huntsville ,
Alabama , Georgetown , Kentucky , Princeton , Indiana , San Antonio , Texas , Buffalo , West Virginia , and Blue Springs , Mississippi . Toyota had a joint -
venture operation with General Motors at New United Motor Manufacturing Inc. in Fremont , California , which began in 1984 and ended in 2009 . It still has a
joint venture with Subaru at Subaru of Indiana Automotive , Inc. in Lafayette , Indiana , which started in 2006 . In these assembly plants , the Camry and the
Tundra are manufactured , among others . Toyota marketing , sales , and distribution in the US are conducted through a separate subsidiary , Toyota Motor Sales
, U.S.A. , Inc . It has started producing larger trucks , such as the new Tundra , to go after the large truck market in the United States . Toyota is also
pushing hybrid vehicles in the US such as the Prius , Camry Hybrid , Highlander Hybrid , and various Lexus products . Currently , Toyota has no plans to offer
diesel motor options in its North American products , including the light - duty pickup trucks .
</item 3>

<item 4>
Toyota
The principle of jidoka , which means the machine stops itself when a problem occurs , became later a part of the Toyota Production System . Looms were built on
a small production line . In 1929 , the patent for the automatic loom was sold to the British company Platt Brothers , generating the starting capital for the
automobile development . Mass production of Toyoda automated loom , displayed at the Toyota Museum in Nagakute - cho , Aichi - gun , Aichi Pref . Japan The
production of Toyota automobiles was started in 1933 as a division of Toyoda Automatic Loom Works devoted to the production of automobiles under the direction
of the founder 's son , Kiichiro Toyoda . Its first vehicles were the A1 passenger car and the G1 in 1935 . The Toyota Motor Co. was established as an
independent company in 1937 Toyoda Standard Sedan AA 1936 Vehicles were originally sold under the name `` Toyoda '' ( トヨダ ) , from the family name of the
company 's founder , Kiichirō Toyoda . In April 1936 , Toyoda 's first passenger car , the Model AA , was completed .
</item 4>

<item 5>
Toyota
Toyota - wikipedia Toyota Jump to : navigation , search For other uses , see Toyota ( disambiguation ) . Toyota Motor Corporation hkl headquarters in Toyota
City , Japan Type Public ( K.K. ) Traded as TYO : 7203 LSE : TYT NYSE : TM TOPIX Core 30 Component Industry Automotive Founded August 28 , 1937 ; 80 years ago (
1937 - 08 - 28 ) Headquarters Toyota , Aichi , Japan Area served Worldwide Key people Kiichiro Toyoda ( Chairman ) Shigeru Hayakawa ( Vice Chairman ) Takeshi
Uchiyamada ( President ) Products Automobiles , luxury vehicles , commercial vehicles , engines Production output 16,985,870 units ( CY 2015 ) Services Banking
, financing , leasing Revenue ¥ 28.40 trillion ( 2016 ) Operating income ¥ 2.85 trillion ( 2016 ) Profit ¥ 2.31 trillion ( 2016 ) Total assets ¥ 47.42 trillion
( 2016 ) Total equity ¥ 17.22 trillion ( 2016 ) Owner Toyota Industries ( 6.73 % ) Nippon Life ( 3.
</item 5>

<item 6>
Toyota
North America ( edit ) Main article : Toyota Motor Engineering & Manufacturing North America Toyota Motor North America headquarters is located in Plano , Texas
, and operates as a holding company in North America . Its manufacturing headquarters is located in Erlanger , Kentucky , and is known as Toyota Motor
Engineering & Manufacturing North America . Toyota Canada Inc. has been in production in Canada since 1983 with an aluminium wheel plant in Delta , British
Columbia , which currently employs a workforce of roughly 260 . Its first vehicle assembly plant , in Cambridge , Ontario , since 1988 , now produces Corolla
compact cars , Matrix crossover vehicles , and Lexus RX 350 luxury SUVs , with a workforce of 4,300 workers . Its second assembly operation in Woodstock ,
Ontario , began manufacturing the RAV4 late in 2008 . In 2006 , Toyota 's subsidiary Hino Motors opened a heavy duty truck plant , also in Woodstock , employing
45 people and producing 2000 trucks annually .
</item 6>

<item 7>
Toyota
However , when Toyota eventually entered the American market in 1957 with the Crown , the name was not well received due to connotations of toys and pets . The
name was soon dropped for the American market , but continued in other markets until the mid-1960s . 1960s -- 1970s ( edit ) By the early 1960s , the US had
begun placing stiff import tariffs on certain vehicles . The so - called `` chicken tax '' of 1964 placed a 25 % tax on imported light trucks . In response to
the tariff , Toyota , Nissan Motor Co. and Honda Motor Co. began building plants in the US by the early 1980s . 1980s ( edit ) With over 40 million sold , the
Corolla is one of the most popular and best selling cars in the world . Toyota received its first Japanese Quality Control Award at the start of the 1980s and
began participating in a wide variety of motorsports . Due to the 1973 oil crisis , consumers in the lucrative US market began turning to small cars with better
fuel economy .
</item 7>

<item 8>
Toyota
In Chinese - speaking countries using simplified Chinese characters ( e.g. China ) , Toyota is known as `` 丰田 '' ( pronounced as Fēngtián in Mandarin Chinese
and Hɔng Tshan in Minnanese ) . These are the same characters as the founding family 's name `` Toyoda '' in Japanese . Logo until 1989 , now still used as a
corporate logo Logo since 1989 Japan ( edit ) Toyota Store Aichi Toyopet Store , Saitama Corolla Store Kyotanabe Netz Store , Tsurumi - ku , Osaka Wikimedia
Commons has media related to Toyota dealerships . In Japan , Toyota currently maintains separate dealership sales channels . The first sales channel established
in 1946 called `` Toyota Store '' ( トヨタ 店 Toyota Mise ) sells large luxury sedans such as the Toyota Century , and the Toyota Crown . In 1955 , the `` Toyopet
Store '' ( トヨペット 店 Toyopetto - ten ) arrived , originally established to sell the Toyota Corona and the Toyopet ToyoAce truck .
</item 8>

<item 9>
Toyota
American car manufacturers had considered small economy cars to be an entry - level product , and their small vehicles employed a low level of quality to keep
the price low . Conservative Toyota held on to rear - wheel - drive designs for longer than most ; while a clear first in overall production they were only
third in production of front - wheel - drive cars in 1983 , behind Nissan and Honda . In part due to this , Nissan 's Sunny managed to squeeze by the Corolla in
numbers built that year . In 1982 , the Toyota Motor Company and Toyota Motor Sales merged into one company , the Toyota Motor Corporation . Two years later ,
Toyota entered into a joint venture with General Motors called the New United Motor Manufacturing , Inc , NUMMI , operating an automobile - manufacturing plant
in Fremont , California . The factory was an old General Motors plant that had been closed for two years . Toyota then started to establish new brands at the
end of the 1980s , with the launch of their luxury division Lexus in 1989 .
</item 9>

<item 10>
Toyota
1990s ( edit ) In the 1990s , Toyota began to branch out from producing mostly compact cars by adding many larger and more luxurious vehicles to its lineup ,
including a full - sized pickup , the T100 ( and later the Tundra ) , several lines of SUVs , a sport version of the Camry , known as the Camry Solara , and the
Scion brand . They would also launch newer and arguably more iconic iterations of their sports cars , namely the MR2 , Celica , and Supra during this era , all
of which have already become icons of the 1990s . With a major presence in Europe , due to the success of Toyota Team Europe , the corporation decided to set up
Toyota Motor Europe Marketing and Engineering , TMME , to help market vehicles in the continent . Two years later , Toyota set up a base in the United Kingdom ,
TMUK , as the company 's cars had become very popular among British drivers . Bases in Indiana , Virginia , and Tianjin were also set up . In 1999 , the company
decided to list itself on the New York and London Stock Exchanges .
</item 10>
=== END ITEM id=3db1bb3a1a3697d3 ===

=== ITEM id=1b562dcabe3699d7 ===
Gold sentences and facts:
Sentence 0: When leaving work in Idaho 's Snake River Valley , Billy Colman sees a pack of dogs attacking a stray coonhound . He takes the stray home to feed it . Once it has rested , Billy sets it free , knowing that it will return home .
  Fact 0.0: Billy Colman leaves work in Idaho's Snake River Valley.
  Fact 0.1: Billy Colman sees a pack of dogs attacking a stray coonhound.
  Fact 0.2: Billy Colman takes the stray coonhound home to feed it.
  Fact 0.3: After the coonhound has rested, Billy Colman sets it free.
  Fact 0.4: Billy Colman knows that the coonhound will return home.

Retrieved items:
<item 1>
Where the Red Fern Grows
Background ( edit ) In a talk given to a group of schoolteachers , Wilson Rawls related how he wrote the first version of the novel ( along with five full
novels , and hundreds of short stories and novelettes ) during the years that he worked on construction in Mexico and Idaho . He rolled the manuscripts up and
saved them in a trunk at his parents ' home . When he met his fiancée , Sophie , he did not want her to know about his failed dreams of becoming a writer , so
about a week before he got married he visited his parents and burned all his manuscripts . He then returned to Idaho and married Sophie . About three months
later , he confessed to his wife that he had burned all his manuscripts and had always dreamed of being a writer . She encouraged him to rewrite one of his
stories . He quit his job and wrote the novel in just three weeks . He said , `` I had it memorized . '' He would not let her read it until it was finished . He
said , `` I finished it on a Friday . I gave it to her Saturday morning and I went to town . I stayed in town all day .
</item 1>

<item 2>
Where the Red Fern Grows
Every year in Tahlequah Oklahoma there is a Red Fern festival . It is listed below in the links . See also ( edit ) List of Where the Red Fern Grows Characters
Footnotes ( edit ) Jump up ^ Where the Red Fern Grows Discussion Guide Scholastic.com ^ Jump up to : Where the Red Fern Grows ( 1974 ) on IMDb Jump up ^ IMDb :
Where the Red Fern Grows : Part Two ^ Jump up to : Wilson Rawls , Dreams Can Come True , audio recording with commentary by Jim Trelease ^ Jump up to :
http://www.trelease-on-reading.com/rawls.html Jump up ^
https://news.google.com/newspapers?nid=336&dat=19740216&id=bMdSAAAAIBAJ&sjid=AH4DAAAAIBAJ&pg=5192,3528282&hl=en Jump up ^ `` All - Time Bestselling Children 's
Books '' . PublishersWeekly.com . Retrieved 2017 - 01 - 29 .
</item 2>

<item 3>
Where the Red Fern Grows
As he tries to call them off , they look at him sadly and he cuts the tree down , which takes him two days and costs him blistered hands . In the end , when he
's about to give up his effort , Billy offers a short prayer for strength to continue . A strong wind starts to blow and the tree falls . Old Dan and Little Ann
take the raccoon down . Billy , Old Dan , and Little Ann go hunting almost every night and become well - known in the Ozarks . Billy and his grandfather make a
bet with Rubin and Rainie Pritchard that Old Dan and Little Ann can tree the `` ghost coon '' . It leads them on a long , complicated hunt , and Rubin and
Rainie want to give up . When they finally have it treed , Billy refuses to kill it . Rubin then orders his dog , Old Blue , to attack Old Dan , who is smaller
; and Rubin starts to beat Billy . Little Ann helps Old Dan , and together they injure Old Blue ; Rubin runs to attack them with an axe , but he trips , falls
on it and kills himself .
</item 3>

<item 4>
Where the Red Fern Grows
Reception ( edit ) Although sales of the novel began slowly , by 1974 over 90,000 copies had been sold . In 2001 , Publishers Weekly estimated that it had sold
6,754,308 copies . Today Where the Red Fern Grows is required reading in many American schools . One critic said it will please adults as well as children . I
remember crying so much through this book , and even today I tear up thinking of Old Dan and Little Ann . I also loaned this to my ( then ) children 's
librarian , because the library copy was always out . I even marked the pages , `` Get out tissue here . '' -- DeAnn Okamura I love , love , love this book with
all my heart and soul . My fourth grade teacher read it to me eons ago , and I 've read it to two of my three boys . There 's something about weeping together
uncontrollably that builds a community of readers ... -- Tess Alfonsin There is a statue of Billy and his dogs at the Idaho Falls Public Library . Every year in
Tahlequah Oklahoma there is a Red Fern festival .
</item 4>

<item 5>
Where the Red Fern Grows
Billy 's grandfather enters him into a championship coon hunt , putting him against experienced hunters and the finest dogs in the country . Before it starts ,
he enters Little Ann into a conformation contest , where she wins the silver cup . On the fourth night of the hunt , Old Dan and Little Ann tree three raccoons
, making it to the final round . The sixth night , they tree one before a blizzard hits . Billy , his father , grandfather , and the judge lose sight of the
dogs . When they finally find them , Billy 's grandfather sprains his ankle . They build a fire , and when Billy 's father chops down a tree , three raccoons
jump out . Old Dan and Little Ann take two of them down , and chase the third to another tree . In the morning , the hunters find them covered with ice circling
the tree . That raccoon wins them the championship , gold cup , and $300 of jackpot money . One night Old Dan and Little Ann tree a mountain lion , and it
attacks . Billy enters the fight with his axe , hoping to save his dogs , but they end up having to save him .
</item 5>

<item 6>
Where the Red Fern Grows
Once it has rested , Billy sets it free , knowing that it will return home . The experience reminds Billy of his childhood in the Ozark Mountains of Oklahoma .
He wants a dog , and his parents offer to get him a mongrel from a neighbor . He wants two coonhounds , but his parents tell him they can not afford them .
Billy finds a magazine ad for a Kentucky kennel which breeds Redbone Coonhounds and sells them for $25 each . He goes to work performing odd jobs and saves the
$50 he needs with the help of his grandfather . Since it takes two years to do so , his grandfather writes to see if the kennel will honor the ad . They do ,
and the price has dropped so the two puppies only cost $40 . Billy 's dogs are delivered to the freight depot in Tahlequah . His family does not have
transportation , so he finds his own way to get to the depot and walks through the hills . He picks up his puppies , which are a male and a female . He buys a
gift for each member of his family with his extra $10 before heading back home .
</item 6>

<item 7>
Where the Red Fern Grows
Where the Red Fern Grows - wikipedia Where the Red Fern Grows Jump to : navigation , search For the film adaptations , see Where the Red Fern Grows ( 1974 film
) and Where the Red Fern Grows ( 2003 film ) . Where the Red Fern Grows First edition hardback cover Author Wilson Rawls Country United States Language English
Genre Children 's novel Publisher Doubleday Publication date 1961 Media type Print ( Hardcover ) Pages 245 pp ISBN 0 - 440 - 22814 - X OCLC 39850615 Where the
Red Fern Grows is a 1961 children 's novel by Wilson Rawls about a boy who buys and trains two Redbone Coonhound hunting dogs . Contents ( hide ) 1 Plot summary
2 Films 3 Background 4 Reception 5 See also 6 Footnotes 7 External links Plot summary ( edit ) When leaving work in Idaho 's Snake River Valley , Billy Colman
sees a pack of dogs attacking a stray coonhound . He takes the stray home to feed it . Once it has rested , Billy sets it free , knowing that it will return
home .
</item 7>

<item 8>
Where the Red Fern Grows
Eventually , they kill the mountain lion , but Old Dan is badly wounded , and Billy finds his intestines in a bush . He dies late that night . Billy is
heartbroken , and Little Ann loses the will to live , stops eating , and dies of grief a few days later on Old Dan 's grave . Billy 's father tries to tell him
that it is all for the best , because with the money they received from winning the championship hunt , they can move to town . He goes to visit Old Dan and
Little Ann 's graves and finds a giant red fern between them . According to Native American legend , only an angel can plant one . He feels ready to move on
knowing that his dogs are always going to be remembered . Films ( edit ) The novel was made into a 1974 film starring Stewart Petersen , James Whitmore ,
Beverly Garland , and Jack Ging . It was followed by a sequel in 1992 , which starred Wilford Brimley , Chad McQueen , Lisa Whelchel , and Karen Carlson . The
film was remade in 2003 and starred Joseph Ashton , Dabney Coleman , Ned Beatty and Dave Matthews .
</item 8>

<item 9>
Where the Red Fern Grows
On the way back home , he spends the night in Robber 's Cave in the Sparrow Hawk Mountains . There he builds a fire and plays with the puppies . While trying to
sleep , he hears a noise that he realizes is the cry of a mountain lion . In the morning , he continues on . He comes to a sycamore tree and sees the names Dan
and Ann carved inside a heart in the bark and decides to name the puppies Old Dan and Little Ann . To train Old Dan and Little Ann , Billy traps a raccoon with
the help of his grandfather and uses its pelt to teach them to hunt . During their training , their personalities become apparent : Old Dan is brave and strong
, while Little Ann is very intelligent . Both are very loyal to each other and to Billy . On the first night of hunting season , Billy takes Old Dan and Little
Ann out for their first hunt . He promises them that if they tree a raccoon , he will do the rest . They tree one in a large sycamore , which Billy had
previously nicknamed `` The Big Tree '' .
</item 9>

<item 10>
Graph query row:
surname: Colman
</item 10>
=== END ITEM id=1b562dcabe3699d7 ===

=== ITEM id=c8ffe30e136f17f6 ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original air date Prod . code U.S. viewers ( millions ) 70 `` The Flash Reborn '' Glen Winter Story by : Andrew Kreisberg Teleplay by : Todd Helbing & Eric Wallace October 10 , 2017 ( 2017 - 10 - 10 ) T27. 13401 2.84 71 `` Mixed Signals '' Alexandra La Roche Jonathan Butler & Gabriel Garza October 17 , 2017 ( 2017 - 10 - 17 ) T27. 13402 2.54 72 `` Luck Be a Lady '' Armen V. Kevorkian Sam Chalsen & Judalina Neira October 24 , 2017 ( 2017 - 10 - 24 ) T27. 13403 2.62 73 `` Elongated Journey Into Night '' Tom Cavanagh Sterling Gates & Thomas Pound October 31 , 2017 ( 2017 - 10 - 31 ) T27. 13404 1.99 74 5 `` Girls Night Out '' Laura Belsey Lauren Certo & Kristen Kim November 7 , 2017 ( 2017 - 11 - 07 ) T27. 13405 2.38 75 6 `` When Harry Met Harry ... '' Brent Crowell Jonathan Butler & Gabriel Garza November 14 , 2017 ( 2017 - 11 - 14 ) T27. 13406 2.46 76 7 `` Therefore I Am '' David McWhirter Eric Wallace & Thomas Pound November 21 , 2017 ( 2017 - 11 - 21 ) T27. 13407 2.20 77 8 `` Crisis on Earth - X , Part 3 '' Dermott Downs Story by : Andrew Kreisberg & Marc Guggenheim Teleplay by : Todd Helbing November 28 , 2017 ( 2017 - 11 - 28 ) T27. 13408 2.82 78 9 `` Do n't Run '' Stefan Pleszczynski Sam Chalsen & Judalina Neira December 5 , 2017 ( 2017 - 12 - 05 ) T27. 13409 2.22 79 10 `` The Trial of The Flash '' Philip Chipera Lauren Certo & Kristen Kim January 16 , 2018 ( 2018 - 01 - 16 ) T27. 13410 2.51 80 11 `` The Elongated Knight Rises '' Alexandra La Roche Sterling Gates & Thomas Pound January 23 , 2018 ( 2018 - 01 - 23 ) T27. 13411 2.12 81 12 `` Honey , I Shrunk Team Flash '' Chris Peppe Sam Chalsen & Judalina Neira January 30 , 2018 ( 2018 - 01 - 30 ) T27. 13412 2.60 82 13 `` True Colors '' Tara Nicole Weyr Jonathan Butler & Gabriel Garza February 6 , 2018 ( 2018 - 02 - 06 ) T27. 13413 2.28 83 14 `` Subject 9 '' Ralph Hemecker Mike Alber & Gabe Snyder February 27 , 2018 ( 2018 - 02 - 27 ) T27. 13414 2.12 84 15 `` Enter Flashtime '' Gregory Smith Todd Helbing & Sterling Gates March 6 , 2018 ( 2018 - 03 - 06 ) T27. 13415 2.04 85 16 `` Run , Iris , Run '' Harry Jierjian Eric Wallace March 13 , 2018 ( 2018 - 03 - 13 ) T27. 13416 2.09 86 17 `` Null and Annoyed '' Kevin Smith Lauren Certo & Kristen Kim April 10 , 2018 ( 2018 - 04 - 10 ) T27. 13417 1.82 87 18 `` Lose Yourself '' Hanelle Culpepper Jonathan Butler & Gabriel Garza April 17 , 2018 ( 2018 - 04 - 17 ) T27. 13418 1.88 88 19 `` Fury Rogue '' Rachel Talalay Joshua V. Gilbert & Jeff Hersh April 24 , 2018 ( 2018 - 04 - 24 ) TBA 1.90 89 20 `` Therefore She Is '' Rob J. Greenlea Sterling Gates & Thomas Pound May 1 , 2018 ( 2018 - 05 - 01 ) TBA TBD 90 21 `` Harry and the Harrisons '' Kevin Mock Judalina Neira & Lauren Certo May 8 , 2018 ( 2018 - 05 - 08 ) TBA TBD 91 22 `` Think Fast '' Viet Nguyen Sam Chalsen & Kristen Kim May 15 , 2018 ( 2018 - 05 - 15 ) TBA TBD 92 23 `` We Are the Flash '' TBA TBA May 22 , 2018 ( 2018 - 05 - 22 ) TBA TBD
  Fact 0.0: Episode 70 overall is titled The Flash Reborn.
  Fact 0.1: The Flash Reborn was directed by Glen Winter.
  Fact 0.2: The Flash Reborn had story by Andrew Kreisberg and teleplay by Todd Helbing and Eric Wallace.
  Fact 0.3: The Flash Reborn aired October 10, 2017, with production code T27.13401 and 2.84 million U.S. viewers.
  Fact 0.4: Episode 71 overall is titled Mixed Signals.
  Fact 0.5: Mixed Signals was directed by Alexandra La Roche.
  Fact 0.6: Mixed Signals was written by Jonathan Butler and Gabriel Garza.
  Fact 0.7: Mixed Signals aired October 17, 2017, with production code T27.13402 and 2.54 million U.S. viewers.
  Fact 0.8: Episode 72 overall is titled Luck Be a Lady.
  Fact 0.9: Luck Be a Lady was directed by Armen V. Kevorkian.
  Fact 0.10: Luck Be a Lady was written by Sam Chalsen and Judalina Neira.
  Fact 0.11: Luck Be a Lady aired October 24, 2017, with production code T27.13403 and 2.62 million U.S. viewers.
  Fact 0.12: Episode 73 overall is titled Elongated Journey Into Night.
  Fact 0.13: Elongated Journey Into Night was directed by Tom Cavanagh.
  Fact 0.14: Elongated Journey Into Night was written by Sterling Gates and Thomas Pound.
  Fact 0.15: Elongated Journey Into Night aired October 31, 2017, with production code T27.13404 and 1.99 million U.S. viewers.
  Fact 0.16: Episode 74 overall is episode 5 of its season and is titled Girls Night Out.
  Fact 0.17: Girls Night Out was directed by Laura Belsey.
  Fact 0.18: Girls Night Out was written by Lauren Certo and Kristen Kim.
  Fact 0.19: Girls Night Out aired November 7, 2017, with production code T27.13405 and 2.38 million U.S. viewers.
  Fact 0.20: Episode 75 overall is episode 6 of its season and is titled When Harry Met Harry ...
  Fact 0.21: When Harry Met Harry ... was directed by Brent Crowell.
  Fact 0.22: When Harry Met Harry ... was written by Jonathan Butler and Gabriel Garza.
  Fact 0.23: When Harry Met Harry ... aired November 14, 2017, with production code T27.13406 and 2.46 million U.S. viewers.
  Fact 0.24: Episode 76 overall is episode 7 of its season and is titled Therefore I Am.
  Fact 0.25: Therefore I Am was directed by David McWhirter.
  Fact 0.26: Therefore I Am was written by Eric Wallace and Thomas Pound.
  Fact 0.27: Therefore I Am aired November 21, 2017, with production code T27.13407 and 2.20 million U.S. viewers.
  Fact 0.28: Episode 77 overall is episode 8 of its season and is titled Crisis on Earth - X, Part 3.
  Fact 0.29: Crisis on Earth - X, Part 3 was directed by Dermott Downs.
  Fact 0.30: Crisis on Earth - X, Part 3 had story by Andrew Kreisberg and Marc Guggenheim and teleplay by Todd Helbing.
  Fact 0.31: Crisis on Earth - X, Part 3 aired November 28, 2017, with production code T27.13408 and 2.82 million U.S. viewers.
  Fact 0.32: Episode 78 overall is episode 9 of its season and is titled Don't Run.
  Fact 0.33: Don't Run was directed by Stefan Pleszczynski.
  Fact 0.34: Don't Run was written by Sam Chalsen and Judalina Neira.
  Fact 0.35: Don't Run aired December 5, 2017, with production code T27.13409 and 2.22 million U.S. viewers.
  Fact 0.36: Episode 79 overall is episode 10 of its season and is titled The Trial of The Flash.
  Fact 0.37: The Trial of The Flash was directed by Philip Chipera.
  Fact 0.38: The Trial of The Flash was written by Lauren Certo and Kristen Kim.
  Fact 0.39: The Trial of The Flash aired January 16, 2018, with production code T27.13410 and 2.51 million U.S. viewers.
  Fact 0.40: Episode 80 overall is episode 11 of its season and is titled The Elongated Knight Rises.
  Fact 0.41: The Elongated Knight Rises was directed by Alexandra La Roche.
  Fact 0.42: The Elongated Knight Rises was written by Sterling Gates and Thomas Pound.
  Fact 0.43: The Elongated Knight Rises aired January 23, 2018, with production code T27.13411 and 2.12 million U.S. viewers.
  Fact 0.44: Episode 81 overall is episode 12 of its season and is titled Honey, I Shrunk Team Flash.
  Fact 0.45: Honey, I Shrunk Team Flash was directed by Chris Peppe.
  Fact 0.46: Honey, I Shrunk Team Flash was written by Sam Chalsen and Judalina Neira.
  Fact 0.47: Honey, I Shrunk Team Flash aired January 30, 2018, with production code T27.13412 and 2.60 million U.S. viewers.
  Fact 0.48: Episode 82 overall is episode 13 of its season and is titled True Colors.
  Fact 0.49: True Colors was directed by Tara Nicole Weyr.
  Fact 0.50: True Colors was written by Jonathan Butler and Gabriel Garza.
  Fact 0.51: True Colors aired February 6, 2018, with production code T27.13413 and 2.28 million U.S. viewers.
  Fact 0.52: Episode 83 overall is episode 14 of its season and is titled Subject 9.
  Fact 0.53: Subject 9 was directed by Ralph Hemecker.
  Fact 0.54: Subject 9 was written by Mike Alber and Gabe Snyder.
  Fact 0.55: Subject 9 aired February 27, 2018, with production code T27.13414 and 2.12 million U.S. viewers.
  Fact 0.56: Episode 84 overall is episode 15 of its season and is titled Enter Flashtime.
  Fact 0.57: Enter Flashtime was directed by Gregory Smith.
  Fact 0.58: Enter Flashtime was written by Todd Helbing and Sterling Gates.
  Fact 0.59: Enter Flashtime aired March 6, 2018, with production code T27.13415 and 2.04 million U.S. viewers.
  Fact 0.60: Episode 85 overall is episode 16 of its season and is titled Run, Iris, Run.
  Fact 0.61: Run, Iris, Run was directed by Harry Jierjian.
  Fact 0.62: Run, Iris, Run was written by Eric Wallace.
  Fact 0.63: Run, Iris, Run aired March 13, 2018, with production code T27.13416 and 2.09 million U.S. viewers.
  Fact 0.64: Episode 86 overall is episode 17 of its season and is titled Null and Annoyed.
  Fact 0.65: Null and Annoyed was directed by Kevin Smith.
  Fact 0.66: Null and Annoyed was written by Lauren Certo and Kristen Kim.
  Fact 0.67: Null and Annoyed aired April 10, 2018, with production code T27.13417 and 1.82 million U.S. viewers.
  Fact 0.68: Episode 87 overall is episode 18 of its season and is titled Lose Yourself.
  Fact 0.69: Lose Yourself was directed by Hanelle Culpepper.
  Fact 0.70: Lose Yourself was written by Jonathan Butler and Gabriel Garza.
  Fact 0.71: Lose Yourself aired April 17, 2018, with production code T27.13418 and 1.88 million U.S. viewers.
  Fact 0.72: Episode 88 overall is episode 19 of its season and is titled Fury Rogue.
  Fact 0.73: Fury Rogue was directed by Rachel Talalay.
  Fact 0.74: Fury Rogue was written by Joshua V. Gilbert and Jeff Hersh.
  Fact 0.75: Fury Rogue aired April 24, 2018, with production code TBA and 1.90 million U.S. viewers.
  Fact 0.76: Episode 89 overall is episode 20 of its season and is titled Therefore She Is.
  Fact 0.77: Therefore She Is was directed by Rob J. Greenlea.
  Fact 0.78: Therefore She Is was written by Sterling Gates and Thomas Pound.
  Fact 0.79: Therefore She Is aired May 1, 2018, with production code TBA and viewer count TBD.
  Fact 0.80: Episode 90 overall is episode 21 of its season and is titled Harry and the Harrisons.
  Fact 0.81: Harry and the Harrisons was directed by Kevin Mock.
  Fact 0.82: Harry and the Harrisons was written by Judalina Neira and Lauren Certo.
  Fact 0.83: Harry and the Harrisons aired May 8, 2018, with production code TBA and viewer count TBD.
  Fact 0.84: Episode 91 overall is episode 22 of its season and is titled Think Fast.
  Fact 0.85: Think Fast was directed by Viet Nguyen.
  Fact 0.86: Think Fast was written by Sam Chalsen and Kristen Kim.
  Fact 0.87: Think Fast aired May 15, 2018, with production code TBA and viewer count TBD.
  Fact 0.88: Episode 92 overall is episode 23 of its season and is titled We Are the Flash.
  Fact 0.89: The director and writers of We Are the Flash are listed as TBA.
  Fact 0.90: We Are the Flash aired May 22, 2018, with production code TBA and viewer count TBD.

Retrieved items:
<item 1>
List of The Flash episodes
13418 1.88 88 19 `` Fury Rogue '' Rachel Talalay Joshua V. Gilbert & Jeff Hersh April 24 , 2018 ( 2018 - 04 - 24 ) TBA 1.90 89 20 `` Therefore She Is '' Rob J.
Greenlea Sterling Gates & Thomas Pound May 1 , 2018 ( 2018 - 05 - 01 ) TBA TBD 90 21 `` Harry and the Harrisons '' Kevin Mock Judalina Neira & Lauren Certo May
8 , 2018 ( 2018 - 05 - 08 ) TBA TBD 91 22 `` Think Fast '' Viet Nguyen Sam Chalsen & Kristen Kim May 15 , 2018 ( 2018 - 05 - 15 ) TBA TBD 92 23 `` We Are the
Flash '' TBA TBA May 22 , 2018 ( 2018 - 05 - 22 ) TBA TBD Ratings ( edit ) Season 1 ( 2014 -- 15 ) ( edit ) No .
</item 1>

<item 2>
List of The Flash episodes
2 / 4 3.14 0.8 1.80 2.0 4.94 10 `` Borrowing Problems From the Future '' January 24 , 2017 1.0 / 3 2.72 0.8 1.92 1.8 4.64 11 `` Dead or Alive '' January 31 ,
2017 1.1 / 4 3.06 0.7 1.64 1.8 4.71 12 `` Untouchable '' February 7 , 2017 1.1 / 4 2.91 0.7 1.68 1.8 4.59 13 `` Attack on Gorilla City '' February 21 , 2017 1.0
/ 4 2.78 0.9 1.96 1.9 4.75 14 `` Attack on Central City '' February 28 , 2017 1.1 / 4 2.87 0.8 1.81 1.9 4.68 15 `` The Wrath of Savitar '' March 7 ,
</item 2>

<item 3>
List of The Flash episodes
13410 2.51 80 11 `` The Elongated Knight Rises '' Alexandra La Roche Sterling Gates & Thomas Pound January 23 , 2018 ( 2018 - 01 - 23 ) T27. 13411 2.12 81 12 ``
Honey , I Shrunk Team Flash '' Chris Peppe Sam Chalsen & Judalina Neira January 30 , 2018 ( 2018 - 01 - 30 ) T27. 13412 2.60 82 13 `` True Colors '' Tara Nicole
Weyr Jonathan Butler & Gabriel Garza February 6 , 2018 ( 2018 - 02 - 06 ) T27. 13413 2.28 83 14 `` Subject 9 '' Ralph Hemecker Mike Alber & Gabe Snyder February
27 , 2018 ( 2018 - 02 - 27 ) T27. 13414 2.12 84 15 `` Enter Flashtime '' Gregory Smith Todd Helbing & Sterling Gates March 6 , 2018 ( 2018 - 03 - 06 ) T27.
</item 3>

<item 4>
List of The Flash episodes
Retrieved February 11 , 2017 . ^ Jump up to : Porter , Rick ( February 1 , 2017 ) . `` ' The Wall ' adjusts up , ' Bones ' adjusts down : Tuesday final ratings
'' . TV by the Numbers . Archived from the original on February 2 , 2017 . Retrieved February 1 , 2017 . ^ Jump up to : Porter , Rick ( February 8 , 2017 ) . ``
' The Middle ' , ' The Wall ' , ' This Is Us ' , FOX comedies all adjust up : Tuesday final ratings '' . TV by the Numbers . Archived from the original on
February 8 , 2017 . Retrieved February 8 , 2017 . ^ Jump up to : Porter , Rick ( February 23 , 2017 ) . `` ' The Flash ' , ' NCIS ' , all others unchanged :
Tuesday final ratings '' . TV by the Numbers . Archived from the original on February 23 , 2017 . Retrieved February 23 , 2017 .
</item 4>

<item 5>
List of The Flash episodes
25 23 October 4 , 2016 ( 2016 - 10 - 04 ) May 23 , 2017 ( 2017 - 05 - 23 ) 120 3.50 23 October 10 , 2017 ( 2017 - 10 - 10 ) May 22 , 2018 ( 2018 - 05 - 22 ) TBA
TBA Episodes ( edit ) Season 1 ( 2014 -- 15 ) ( edit ) No . overall No. in season Title Directed by Written by Original air date Prod . code U.S. viewers (
millions ) `` Pilot '' David Nutter Story by : Greg Berlanti & Andrew Kreisberg & Geoff Johns Teleplay by : Andrew Kreisberg & Geoff Johns October 7 , 2014 (
2014 - 10 - 07 ) 296648 4.83 Barry Allen , a forensic crime scene assistant for Central City Police Department , is struck by lightning during a storm that is
created after a particle accelerator from S.T.A.R. Labs explodes . In a coma for nine months , Barry awakens in S.T.A.R.
</item 5>

<item 6>
List of The Flash episodes
13401 2.84 71 `` Mixed Signals '' Alexandra La Roche Jonathan Butler & Gabriel Garza October 17 , 2017 ( 2017 - 10 - 17 ) T27. 13402 2.54 72 `` Luck Be a Lady
'' Armen V. Kevorkian Sam Chalsen & Judalina Neira October 24 , 2017 ( 2017 - 10 - 24 ) T27. 13403 2.62 73 `` Elongated Journey Into Night '' Tom Cavanagh
Sterling Gates & Thomas Pound October 31 , 2017 ( 2017 - 10 - 31 ) T27. 13404 1.99 74 5 `` Girls Night Out '' Laura Belsey Lauren Certo & Kristen Kim November 7
, 2017 ( 2017 - 11 - 07 ) T27. 13405 2.38 75 6 `` When Harry Met Harry ... '' Brent Crowell Jonathan Butler & Gabriel Garza November 14 , 2017 ( 2017 - 11 - 14
) T27.
</item 6>

<item 7>
List of The Flash episodes
08 1.0 2.20 2.3 6.28 12 `` Crazy for You '' February 3 , 2015 1.3 / 4 3.60 0.9 N / A 2.2 N / A 13 `` The Nuclear Man '' February 10 , 2015 1.5 / 5 3.66 N / A N
/ A N / A N / A 14 `` Fallout '' February 17 , 2015 1.5 / 4 4.01 0.9 N / A 2.4 N / A 15 `` Out of Time '' March 17 , 2015 1.3 / 5 3.69 1.0 2.34 2.3 6.09 16 ``
Rogue Time '' March 24 , 2015 1.2 / 4 3.33 1.0 2.37 2.2 5.69 17 `` Tricksters '' March 31 , 2015 1.3 / 4 3.67 1.0 2.43 2.
</item 7>

<item 8>
List of The Flash episodes
13406 2.46 76 7 `` Therefore I Am '' David McWhirter Eric Wallace & Thomas Pound November 21 , 2017 ( 2017 - 11 - 21 ) T27. 13407 2.20 77 8 `` Crisis on Earth -
X , Part 3 '' Dermott Downs Story by : Andrew Kreisberg & Marc Guggenheim Teleplay by : Todd Helbing November 28 , 2017 ( 2017 - 11 - 28 ) T27. 13408 2.82 78 9
`` Do n't Run '' Stefan Pleszczynski Sam Chalsen & Judalina Neira December 5 , 2017 ( 2017 - 12 - 05 ) T27. 13409 2.22 79 10 `` The Trial of The Flash '' Philip
Chipera Lauren Certo & Kristen Kim January 16 , 2018 ( 2018 - 01 - 16 ) T27.
</item 8>

<item 9>
List of The Flash episodes
13415 2.04 85 16 `` Run , Iris , Run '' Harry Jierjian Eric Wallace March 13 , 2018 ( 2018 - 03 - 13 ) T27. 13416 2.09 86 17 `` Null and Annoyed '' Kevin Smith
Lauren Certo & Kristen Kim April 10 , 2018 ( 2018 - 04 - 10 ) T27. 13417 1.82 87 18 `` Lose Yourself '' Hanelle Culpepper Jonathan Butler & Gabriel Garza April
17 , 2018 ( 2018 - 04 - 17 ) T27.
</item 9>

<item 10>
List of The Flash episodes
9 4.68 15 `` The Wrath of Savitar '' March 7 , 2017 0.9 / 3 2.52 0.7 1.71 1.6 4.23 16 `` Into the Speed Force '' March 14 , 2017 0.9 / 3 2.39 0.7 1.70 1.6 4.10
17 `` Duet '' March 21 , 2017 1.0 / 4 2.71 0.8 1.88 1.8 4.59 18 `` Abra Kadabra '' March 28 , 2017 0.9 / 3 2.39 0.7 1.67 1.6 4.06 19 `` The Once and Future
Flash '' April 25 , 2017 1.0 / 4 2.67 0.9 1.90 1.9 4.57 20 `` I Know Who You Are '' May 2 , 2017 1.0 / 4 2.69 0.8 1.
</item 10>
=== END ITEM id=c8ffe30e136f17f6 ===

=== ITEM id=42a1137c4c77ca7d ===
Gold sentences and facts:
Sentence 0: Starting at 500 million Population ( in billions ) 0.5 8 Year 1500 1804 1927 2024 Years elapsed 304 123 47 50
  Fact 0.0: The starting value is 500 million.
  Fact 0.1: The population values, in billions, are 0.5 and 8.
  Fact 0.2: The year values are 1500, 1804, 1927 and 2024.
  Fact 0.3: The years elapsed values are 304, 123, 47 and 50.

Retrieved items:
<item 1>
World population
The vertical axis is logarithmic and is in millions of people . Population size fluctuates at differing rates in differing regions . Nonetheless , population
growth is the long - standing trend on all inhabited continents , as well as in most individual states . During the 20th century , the global population saw its
greatest increase in known history , rising from about 1.6 billion in 1900 to over 6 billion in 2000 . A number of factors contributed to this increase ,
including the lessening of the mortality rate in many countries by improved sanitation and medical advances , and a massive increase in agricultural
productivity attributed to the Green Revolution . In 2000 , the United Nations estimated that the world 's population was growing at an annual rate of 1.14 % (
equivalent to around 75 million people ) , down from a peak of 88 million per year in 1989 . By 2000 , there were approximately ten times as many people on
Earth as there had been in 1700 .
</item 1>

<item 2>
World population
The world population has grown by over four billion since the beginning of the Green Revolution , but food production has so far kept pace with population
growth . Most scholars believe that , without the Revolution , there would be greater levels of famine and malnutrition than the UN presently documents .
However , neo-Malthusians point out that fossil fuels provided the energy for the Green Revolution , in the form of natural gas - derived fertilizers , oil -
derived pesticides , and hydrocarbon - fueled irrigation , and that many crops have become so genetically uniform that a crop failure in any one country could
potentially have global repercussions . In 2004 , a meta - analysis of 70 quantitative studies estimating a sustainable limit to the world population generated
a meta - estimate of 7.7 billion people . Graph of the global human population from 10,000 BC to AD 2000 , from the US Census Bureau . The graph shows the
extremely rapid growth in the world population that has taken place since the 18th century . The human population has been growing rapidly since the mid-1800s .
</item 2>

<item 3>
World population
Projected figures vary depending on underlying statistical assumptions and the variables used in projection calculations , especially the fertility variable .
Long - range predictions to 2150 range from a population decline to 3.2 billion in the `` low scenario '' , to `` high scenarios '' of 24.8 billion . One
extreme scenario predicted a massive increase to 256 billion by 2150 , assuming the global fertility rate remained at its 1995 level of 3.04 children per woman
; however , by 2010 the global fertility rate had declined to 2.52 . There is no estimation for the exact day or month the world 's population surpassed one or
two billion . The points at which it reached three and four billion were not officially noted , but the International Database of the United States Census
Bureau placed them in July 1959 and April 1974 respectively . The United Nations did determine , and commemorate , the `` Day of 5 Billion '' on July 11 , 1987
, and the `` Day of 6 Billion '' on October 12 , 1999 . The Population Division of the United Nations declared the `` Day of 7 Billion '' to be October 31 ,
2011 .
</item 3>

<item 4>
World population
Milestones by the billions ( edit ) Main article : World population milestones World population milestones in billions Population 5 6 7 8 9 Year 1804 1927 1960
1999 2011 2024 2042 Years elapsed -- -- 123 33 14 13 12 12 13 18 It is estimated that the world population reached one billion for the first time in 1804 . It
was another 123 years before it reached two billion in 1927 , but it took only 33 years to reach three billion in 1960 . Thereafter , the global population
reached four billion in 1974 , five billion in 1987 , six billion in 1999 and , according to the United States Census Bureau , seven billion in March 2012 . The
United Nations , however , estimated that the world population reached seven billion in October 2011 . According to current projections , the global population
will reach eight billion by 2024 , and will likely reach around nine billion by 2042 . Alternative scenarios for 2050 range from a low of 7.4 billion to a high
of more than 10.6 billion .
</item 4>

<item 5>
World population
758 254 1,534 577 193 187 14.2 1960 3,018 285 1,687 606 221 204 15.8 1965 3,322 322 1,875 635 254 219 17.5 1970 3,682 366 2,120 657 288 231 19.7 4,061 416 2,378
677 326 242 21.5 1980 4,440 478 2,626 694 365 254 23.0 1985 4,853 550 2,897 708 406 267 24.9 5,310 632 3,202 721 447 281 27.0 1995 5,735 720 3,475 728 487 296
29.1 2000 6,127 814 3,714 726 527 314 31.1 2005 6,520 920 3,945 729 564 329 33.4 6,930 1,044 4,
</item 5>

<item 6>
World population
The transition from hyperbolic growth to slower rates of growth is related to the demographic transition . According to the Russian demographer Sergey Kapitsa ,
the world population grew between 67,000 BC and 1965 according to the following formula : N = C τ arccot ⁡ T 0 − T τ ( \ displaystyle N = ( \ frac ( C ) ( \ tau
) ) \ operatorname ( arccot ) ( \ frac ( T_ ( 0 ) - T ) ( \ tau ) ) ) where N is current population T is the current year C = ( 1.86 ± 0.01 ) 10 T = 2007 ± 1 τ
( \ displaystyle \ tau ) = 42 ± 1 Years for world population to double ( edit ) According to linear interpolation and extrapolation of UNDESA population
estimates , the world population has doubled , or will double , in the years listed in the tables below ( with two different starting points ) . During the 2nd
millennium , each doubling took roughly half as long as the previous doubling , fitting the hyperbolic growth model mentioned above .
</item 6>

<item 7>
World population
Improved material conditions led to the population of Britain increasing from 10 million to 40 million in the 19th century . The population of the United
Kingdom reached 60 million in 2006 . The United States saw its population grow from around 5.3 million in 1800 to 106 million in 1920 , exceeding 307 million in
2010 . The first half of the 20th century in Imperial Russia and the Soviet Union was marked by a succession of major wars , famines and other disasters which
caused large - scale population losses ( approximately 60 million excess deaths ) . After the collapse of the Soviet Union , Russia 's population declined
significantly -- from 150 million in 1991 to 143 million in 2012 -- but by 2013 this decline appeared to have halted . Many countries in the developing world
have experienced extremely rapid population growth since the early 20th century , due to economic development and improvements in public health . China 's
population rose from approximately 430 million in 1850 to 580 million in 1953 , and now stands at over 1.3 billion .
</item 7>

<item 8>
World population
Archived from the original on May 11 , 2009 . Retrieved 2013 - 07 - 26 . Jump up ^ Lutz , Wolfgang ; Sanderson , Warren ; Scherbov , Sergei ( 1997 - 06 - 19 ) .
`` Doubling of world population unlikely '' . Nature . 387 ( 6635 ) : 803 -- 805 . doi : 10.1038 / 42935 . PMID 9194559 . Retrieved 2012 - 10 - 23 . ^ Jump up
to : `` World faces ' perfect storm ' of problems by 2030 , chief scientist to warn '' . The Guardian . March 18 , 2009 . Retrieved February 18 , 2013 . Jump up
^ Jowit , Juliette ( October 23 , 2011 ) . `` Paul Ehrlich , a prophet of global population doom who is gloomier than ever '' . The Guardian . London .
Retrieved July 22 , 2013 .
</item 8>

<item 9>
World population
However , after 2024 , it is unlikely that there will be another doubling of the global population in the 21st century . Historic chart showing the periods of
time the world population has taken to double , from 1700 to 2000 Starting at 500 million Population ( in billions ) 0.5 8 Year 1500 1804 1927 2024 Years
elapsed 304 123 47 50 Starting at 375 million Population ( in billions ) 0.375 0.75 1.5 6 Year 1171 1715 1881 1960 1999 Years elapsed 544 166 79 39
Overpopulation ( edit ) Main articles : Human overpopulation and Malthusian catastrophe Predictions of scarcity ( edit ) Greater Los Angeles lies on a coastal
mediterranean savannah with a small watershed that is able to support at most one million people on its own water ; as of 2015 , the area has a population of
over 18 million . Researchers predict that similar cases of resource scarcity will grow more common as the world population increases .
</item 9>

<item 10>
World population
000 BC < 0.015 0 0 10,000 BC 8000 BC 5 6500 BC 5 5000 BC 5 4000 BC 7 3000 BC 14 2000 BC 27 1000 BC 50 7 33 9 500 BC 100 14 66 16 AD 1 200 23 141 28 1000 400 70
269 50 8 1500 458 86 243 84 39 1600 580 114 339 111 10 1700 682 106 436 125 10 1750 791 106 502 163 16 1800 1,000 107 656 203 24 7 1850 1,262 111 809 276 38 26
1900 1,650 133 947 408 74 82 6 1950 2,525 229 1,394 549 169 172 12.7 1955 2,758 254 1,534 577 193 187 14.
</item 10>
=== END ITEM id=42a1137c4c77ca7d ===
