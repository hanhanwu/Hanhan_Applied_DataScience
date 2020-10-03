# Meeting Notes with Stakeholders

## Rules of Thumb
* Double check the action items at the end of the meeting with them
  * May need to put priority in the items
* Company roadmap change
  * The roadmap change might affect the timeline and plan of each project

## Technical Stakeholders
### Roles
* Architects, Dev Managers
### Use Cases
#### Build a new solution for an existing solution
* This can be rather solution replace or the modification of the existing solution.
* Suggestions
  * Understand existing Prod architect, available data sources, output, how are the results shown to the clients
  * Understand from the very beginning that the new solution will be a new build on Prod side or needs to use some existing Prod solution
    * In order to avoid the conflicts appeared in later prod deployment
  * Address conflicts between the new solution and existing solution if in evitable
    * If there is conflicts, think about how to help Dev teams to address that, mention that when discussing with the stakeholders
    * Or have frequent discussion with the stakeholders or tech people in charge, to resolve more conflicts when builging the new solution
  * Know anything available at Prod side but not available for DS, if it's worthy to use, work with PMs to get the data flow for DS
    * If there could be data version issues of the data can get, understand the differneces between data versions and measure the risks